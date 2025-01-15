# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns  # 可选，用于美化热力图
import numpy as np

# from tokenizer_me.tokenizer_image.vq_model import VQ_models
from tokenizer_me.tokenizer_image.vq_model_sw import VQ_models

from autoregressive.models.gpt_multipos import GPT_models
from autoregressive.infer.generate_diagonal import generate
from autoregressive.models.meta_learner import MetaLearner, ModelArgs

# TODO: attention score whether still sinks
def visualize_fused_attention_maps_v2(model, method = "log_normalize", save_to_disk=False, output_dir="fused_attention_maps"):
    """
    可视化逐步生成情况下每层融合所有 Head 的 Attention Map。

    参数：
    - model: 训练好的 Transformer 模型。
    - seq_len: 输入序列的长度。
    - save_to_disk: 是否保存图片到磁盘。
    - output_dir: 保存文件的目录。
    """
    import os

    if save_to_disk:
        os.makedirs(output_dir, exist_ok=True)

    # 遍历每层的 Attention Map
    for layer_idx, fused_map in enumerate(model.fused_attention_maps):
        if fused_map is None:
            print(f"Warning: No Attention Map for Layer {layer_idx}")
            continue

        # 如果是 bfloat16，则先转换为 float32
        if fused_map.dtype == torch.bfloat16:
            fused_map = fused_map.to(torch.float32)

        # 转为 NumPy 格式并归一化
        fused_map = fused_map.detach().cpu().numpy()
        # 对比度增强处理
        if method == "log":
            fused_map = np.log1p(fused_map)  # 对数变换
        elif method == "normalize":
            fused_map = (fused_map - fused_map.min()) / (fused_map.max() - fused_map.min())  # 归一化
        elif method == "log_normalize":
            fused_map = (fused_map - fused_map.min()) / (fused_map.max() - fused_map.min())  # 归一化
            fused_map = np.log1p(fused_map)  # 再进行对数变换
        else:
            raise ValueError(f"Unsupported method: {method}")

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(fused_map, cmap="viridis", cbar=True)
        plt.title(f"Fused Attention Map - Layer {layer_idx}")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")

        if save_to_disk:
            filename = f"{output_dir}/fused_layer_{layer_idx}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    meta_learner = MetaLearner(ModelArgs, gpt_dim=gpt_model.dim).to(device=device, dtype=precision)
    # meta_learner = MetaLearner(ModelArgs, gpt_dim=gpt_model.dim).to(device=device)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:  # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
        meta_learner_weight = checkpoint["meta_learner"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
        meta_learner_weight = checkpoint["meta_learner"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
        meta_learner_weight = checkpoint["meta_learner"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    meta_learner.load_state_dict(meta_learner_weight, strict=False)
    meta_learner.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )  # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo")

        # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # class_labels = [780, 780, 780, 780, 780, 780, 780, 780, ]
    # class_labels = [77, 77, 77, 77, 77, 77, 77, 77, ]
    # class_labels = torch.randint(1,1000, (8,), dtype=torch.long)
    print(class_labels)
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

    t1 = time.time()
    index_sample = generate(
        gpt_model, meta_learner, c_indices, latent_size ** 2,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True,
    )
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")

    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # visualize_fused_attention_maps_v2(gpt_model, save_to_disk=False)

    # Save and display images:
    save_image(samples, "sample_{}.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str,
                        # default=r"C:\Users\谷涵溪\gitproj\LlamaGen/pretrained_models/c2i_L_384.pt")
                        # default=r"D:\Desktop\SHU\Intern\AILAb\MINDmap\autoregressive\train\results\0040000.pt")
                        default=r"D:\Desktop\SHU\Intern\AILAb\Next_Diagonal\autoregressive\models\cache\0040000.pt")

    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i",
                        help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--vq-ckpt", type=str,
                        default=r"D:\Desktop\SHU\Intern\AILAb\MINDmap\tokenizer_me\tokenizer_image\cache\ckpt0_swin100M_IN1000.pth",
                        help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512,224], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)
