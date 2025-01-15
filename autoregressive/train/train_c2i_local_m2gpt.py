# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse

from utils_me.logger import create_logger, create_local_logger
from utils_me.distributed import init_distributed_mode
from utils_me.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt_multipos import GPT_models

from autoregressive.infer.generate_diagonal import get_diagonal_indices
from autoregressive.models.meta_learner import MetaLearner, ModelArgs
from tokenizer_me.tokenizer_image.vq_model_sw import VQ_models


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save_image:
        from torchvision.utils import save_image
        # create and load model
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.vocab_size,
            codebook_embed_dim=8, )
        vq_model.to(device)
        vq_model.eval()
        vq_checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(vq_checkpoint["model"] if "model" in vq_checkpoint else vq_checkpoint)
        del vq_checkpoint
        print(f"image tokenizer is loaded")

    # Setup DDP:

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_local_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
    cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
    os.makedirs(cloud_checkpoint_dir, exist_ok=True)
    logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")

    # training args
    logger.info(f"{args}")

    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)

    block_size = args.image_size // args.downsample_size

    # with torch.device(device):
    #     # 为模型设置缓存以匹配最大批次大小和序列长度
    #     model.setup_caches(max_batch_size=args.global_batch_size*2, max_seq_length=block_size**2,
    #                        dtype=model.tok_embeddings.weight.dtype)
    # causal_mask = create_diagonal_causal_mask(block_size ** 2, args.global_batch_size * 2, block_size ** 2)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    meta_learner = MetaLearner(ModelArgs, gpt_dim=model.dim).to(device)
    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    optimizer_meta = creat_optimizer(meta_learner, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # if not args.no_compile:
    #     logger.info("compiling the model... (may take several minutes)")
    #     model = torch.compile(model)  # requires PyTorch 2.0

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    meta_learner.train()  # important! This enables embedding dropout for classifier-free guidance

    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # Initialize gradient scaler for mixed precision training
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    linear_positions = get_diagonal_indices(block_size)  # [(r,c,pos),...]

    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    loss_history = []

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")

        for x, y in loader:
            x = x.permute(0, 2, 1, 3).reshape(-1, block_size ** 2)
            y = y.repeat(1, 2).squeeze(0)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            batch_size = z_indices.size(0)
            seq_len = z_indices.size(1)
            assert seq_len == block_size * block_size

            z_indices_2d = z_indices.view(batch_size, block_size, block_size)

            # Initialize lists to hold all lefts, ups, and targets for all diagonals
            all_lefts = []
            all_ups = []
            targets = []
            targets.append(z_indices_2d[:, 0, 0])

            # Collect all positions idx at once for efficiency
            for i, diag in enumerate(linear_positions[1:], start=1):
                for (r, c, p) in diag:
                    diag_len = len(diag)
                    left_idxes = z_indices_2d[:, r, c - 1] if c > 0 else (model.vocab_size + 1) * torch.ones_like(
                        z_indices[:, 0],
                        dtype=torch.long,
                        device=device)  # set the <left bond>
                    up_idxes = z_indices_2d[:, r - 1, c] if r > 0 else (model.vocab_size + 2) * torch.ones_like(
                        z_indices[:, 0],
                        dtype=torch.long,
                        device=device)  # set the <up bond>
                    all_lefts.append(left_idxes)
                    all_ups.append(up_idxes)
                    targets.append(z_indices_2d[:, r, c])

            # Stack everything together after collecting all diagonals
            lefts_batch = torch.stack(all_lefts, dim=1)  # [batch_size, total_diag_len]
            ups_batch = torch.stack(all_ups, dim=1)  # [batch_size, total_diag_len]
            targets_batch = torch.stack(targets, dim=1)  # [batch_size, total_diag_len]
            # print(targets_batch.shape)

            # Pass all data to MetaLearner at once
            with torch.cuda.amp.autocast(dtype=ptdtype):  # Ensure you're using mixed precision if applicable
                idx_embeddings = meta_learner(lefts_batch, ups_batch,
                                              train=True)  # [batch_size, total_diag_len, gpt_dim]
                # print(idx_embeddings.shape)

                _, loss = model(cond_idx=c_indices, idx_embedding=idx_embeddings[:, :], targets=targets_batch)

            # print(f"Diag Time cost = {end_time - start_time:.3f}, ")

            # Backward pass
            scaler.scale(loss).backward()

            avg_loss = loss.item()
            scaler.scale(avg_loss).backward()
            # end_time = time.time()
            # print(f"Batch Avg Loss: {avg_loss}, time cost = {end_time - start_time:.3f}, ")

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                scaler.unscale_(optimizer_meta)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            # step the optimizer and scaler if training in fp16
            optimizer.step()
            optimizer_meta.step()
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            optimizer_meta.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += avg_loss
            log_steps += 1
            train_steps += 1

            loss_history.append(avg_loss)

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                model_weight = model.state_dict()
                meta_learner_weight = meta_learner.state_dict()
                checkpoint = {
                    "model": model_weight,
                    "meta_learner": meta_learner_weight,
                    "optimizer": optimizer.state_dict(),
                    "optimizer_meta": optimizer_meta.state_dict(),
                    "steps": train_steps,
                    "args": args
                }

                if args.ema:
                    checkpoint["ema"] = ema.state_dict()
                if not args.no_local_save:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, cloud_checkpoint_path)
                logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

            if args.save_image and train_steps % args.log_every == 0:
                from autoregressive.infer.generate_diagonal import generate
                with torch.no_grad():
                    class_labels = torch.randint(1, 1000, (16,), dtype=torch.long)
                    c_indices = torch.tensor(class_labels, device=device)
                    qzshape = [len(class_labels), 8, latent_size, latent_size]
                    t1 = time.time()
                    index_sample = generate(
                        model, meta_learner, c_indices, latent_size ** 2,
                        cfg_scale=1., cfg_interval=-1,
                        temperature=1., top_k=1000,
                        top_p=1., sample_logits=True,
                    )
                    sampling_time = time.time() - t1
                    t2 = time.time()
                    samples = vq_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]
                decoder_time = time.time() - t2

                # Save and display images:
                save_image(samples, f"{cloud_checkpoint_dir}/sample_{epoch}_{args.gpt_model}.png", nrow=4,
                           normalize=True, value_range=(-1, 1))
                logger.info(
                    f"(class = {class_labels}) Generation Sec: {steps_per_sec:.2f}, image is saved to sample_{train_steps}_{args.gpt_model}.png, gpt sampling takes about {sampling_time:.2f} seconds. Decoder takes about {decoder_time:.2f} seconds.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--vq-ckpt", type=str,
                        default=r"D:\Desktop\SHU\Intern\AILAb\MINDmap\tokenizer_me\tokenizer_image\cache\ckpt0_swin100M_IN1000.pth",
                        help="ckpt path for vq model")
    parser.add_argument("--save-image", default=True, type=bool, required=False)
    parser.add_argument("--code-path", default=r"./data/tmp", type=str, required=False)
    parser.add_argument("--cloud-save-path", default="./log/cloud", type=str, required=False,
                        help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true',
                        help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i",
                        help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512, 224], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=14)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    args = parser.parse_args()
    main(args)
