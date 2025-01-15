import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_me.utils import get_diagonal_indices


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    # TODO: to find whether it is the sample problem and whether the stochastic probability in the genration tokens means?
    # n = 4
    # threshold = logits.max(dim=-1, keepdim=True).values - n * logits.std(dim=-1, keepdim=True)
    # logits[logits < threshold] = float('-inf')

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float = 1., top_k: int = 1000, top_p: float = 1.0, sample_logits=True):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def generate_diagonals(
        model, meta_learner, prefix_len: int,
        cur_tokens: torch.Tensor, block_size: int, cfg_scale: float, cfg_interval: int,
        **sampling_kwargs):
    """
    Input: the left-top token, Output: the overall sequence [1D]
    Generation with the time complexity of O(n) = 2n
    Generation from the left-top, by prefill, then (h, w)
    =========Latent Index=========
    [1, 2, 4, ...]      [(0, 0), (0, 1), (0, 2), ...]
    [3, 5, ...]         [(1, 0), (1, 1), ...]
    [6, ...]            [(2, 0), ...]
    ...                 ...
    =========Latent Index=========
    BATCH I: (0, 0) --> 1,
    BATCH II: [0...0, 1] --> (0, 1), [1, 0...0] --> (1, 0) || (0, 1) --> 2, (1, 0) --> 3,
    BATCH III: [0...0, 3] --> (2, 0), [3, 2] --> (1, 1), [2, 0...0] --> (0, 2), || (0, 2) --> 4, (1, 1) --> 5, (2, 0) --> 6,
    ...up to batch XVI, with filling "0" to the left of top margin. After XVI, there's no need.
    """
    cfg_flag = True
    linear_positions = get_diagonal_indices(block_size)
    print("Block size is: ", block_size, "Prefix length is: ", prefix_len)
    print("Generation sequence in diagonal flattened", linear_positions)
    device = cur_tokens.device

    batch_size = cur_tokens.size(0)

    # first token is generated by the prefill with class embedding：
    final_tokens = [[None for _ in range(block_size)] for _ in range(block_size)]
    final_tokens[0][0] = cur_tokens[:, 0]
    cur_max = 1

    for diag in linear_positions[1:]:
        diag_idx_embeddings = []
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False,
                                            enable_math=True):  # Actually better for Inductor to codegen attention here
            if cfg_interval > -1:
                cfg_flag = False

            current_input_pos = torch.tensor([p + prefix_len for (_, _, p) in diag], device=device)
            """Ensure the length fits the generation"""
            assert current_input_pos.shape[-1] == cur_max + 1 if cur_max < block_size else 2 * block_size - cur_max - 1

            for idx_in_diag, (r, c, pos) in enumerate(diag):
                # print(r, c, pos, cur_tokens.shape)  # if needed to better sequences' visualization.
                left_idx = (model.vocab_size + 1) * torch.ones_like(cur_tokens[:, 0],
                                                                    dtype=torch.long,
                                                                    device=device)  # set the <left bond>
                up_idx = (model.vocab_size + 2) * torch.ones_like(cur_tokens[:, 0],
                                                                  dtype=torch.long,
                                                                  device=device)  # set the <left bond>
                if c > 0:
                    left_idx = cur_tokens[:, idx_in_diag - 1]
                if r > 0:
                    up_idx = cur_tokens[:, idx_in_diag]
                idx_embeddings = meta_learner(left_idx, up_idx)
                # print(idx_embeddings.shape) # [B, hidden_dim]
                diag_idx_embeddings.append(idx_embeddings)
            diag_idx_embeddings = torch.stack(diag_idx_embeddings, dim=1)
            # print(idx_embeddings.shape) # [B, num_pos, ,hidden_dim]
            print("Current diagonal len: ", current_input_pos.shape)

            if cfg_scale > 1.0:
                logits, _ = model(diag_idx_embeddings, cond_idx=None, input_pos=current_input_pos)
                logits_combined = logits
                cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
                if cfg_flag:
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits = cond_logits
            else:
                logits, _ = model(diag_idx_embeddings, cond_idx=None, input_pos=current_input_pos)

            logits_merged = logits.view(-1, logits.size(-1))  # [diag_len*batch, 16384]
            # print(logits_merged.shape, current_input_pos.shape)
            logits_merged = torch.unsqueeze(logits_merged, dim=1)
            cur_tokens, _ = sample(logits_merged, **sampling_kwargs)
            # print(cur_tokens.shape)
            original_shape = (logits.size(0), logits.size(1))
            cur_tokens = cur_tokens.view(original_shape)  # [2, 2]
            for idx_in_diag, (r, c, pos) in enumerate(diag):
                final_tokens[r][c] = cur_tokens[:, idx_in_diag]

            cur_max += 1

    result = torch.stack([torch.stack(row) for row in final_tokens])
    result = result.squeeze(-1).permute(2, 0, 1)

    return result


@torch.no_grad()
def generate(model, meta_learner, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1,
             **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1  # 'c2i prefix'

    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]  # 't2i prefix'
    else:
        raise Exception("please check model type")
    max_batch_size = cond.shape[0]
    T_new = T + max_new_tokens
    max_seq_length = T_new  # [prefix_len + block_size, ]
    device = cond.device
    print("Overall sequence length: ", max_seq_length)
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length,
                           dtype=model.tok_embeddings.weight.dtype)

    if emb_masks is not None:  # TODO
        """
        None for now
        """
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

    # position for the first token
    input_pos = torch.arange(1, T + 1, device=device)
    print(input_pos)
    # get the (r, c, p) = (0, 0, 0)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    print(next_token.shape, max_new_tokens)
    DD_seq = generate_diagonals(model, meta_learner, prefix_len=T, cur_tokens=next_token,
                                block_size=int(max_new_tokens ** 0.5),
                                cfg_scale=cfg_scale, cfg_interval=cfg_interval, **sampling_kwargs)
    """[B, block_size, , block_size]"""
    DD_seq = DD_seq.reshape(max_batch_size, -1)
    return DD_seq
