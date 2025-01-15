# Modified from original GPT implementation
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils_me.drop_path import DropPath
from utils_me.utils import get_diagonal_indices, create_diagonal_causal_mask, generate_reverse_mapping

import os, math, random

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


def find_multiple(n: int, k: int):
    """
    Try to makeup for the not mode length of latent sequences.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    class space token: <num_classes + 1> indicating using CFG
    set (force_drop_ids != None) or (train or Dropout == True) to utilize CFG
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        if force_drop_ids is None,
            drop labels as the probability of self.dropout_prob

        param: label: [batch_size, ]
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S, ]
        # k_val: [B, H, S, D]，S means input osition
        # write multi-position kv into buffer
        # input_pos should not have same elements

        self.k_cache[:, :, input_pos, :] = k_val
        self.v_cache[:, :, input_pos, :] = v_val

        return self.k_cache[:, :, :input_pos[-1] + 1, :], self.v_cache[:, :, :input_pos[-1] + 1, :]


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (
                               self.n_head + 2 * self.n_kv_head) * self.head_dim  # most times it's 3 * self.n_head *  self.head_dim

        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
            self, x: torch.Tensor,
            freqs_cis: torch.Tensor = None,
            input_pos: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ):
        # x: [B, S, D]
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        # xq, xk, xv: [B, S, n_heads1, head_dim]

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None and input_pos is not None:
            # input_pos: [S_multi-position, ]
            keys, values = self.kv_cache.update(input_pos, xk, xv)
            mask = mask[:, :, :, :input_pos[-1] + 1]
        else:
            keys, values = xk, xv

        # makes
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        # print(xq.shape, keys.shape, values.shape, input_pos, mask.shape)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=(mask is None),
            dropout_p=self.attn_dropout_p if self.training else 0
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention_weights = None

    def forward(
            self, x: torch.Tensor, freqs_cis: torch.Tensor,
            input_pos: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ):
        # Normalize and compute attention
        normed_x = self.attention_norm(x)
        attention_output = self.attention(
            normed_x,
            freqs_cis=freqs_cis,
            input_pos=input_pos,
            mask=mask
        )

        # Apply residual connection and drop path
        h = x + self.drop_path(attention_output)

        # Apply feed forward layer with residual connection
        normed_h = self.ffn_norm(h)
        ffn_output = self.feed_forward(normed_h)
        out = h + self.drop_path(ffn_output)

        return out


class Transformer(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head,
                                                 self.config.rope_base, self.cls_token_num, mode="diagonal")
        # self.freqs_cis = precompute_freqs_cis(grid_size**2, self.config.dim // self.config.n_head,
        #                                          self.config.rope_base, self.cls_token_num)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

        self.fused_attention_maps = [None] * len(self.layers)  # TODO visualization of attention score

        self.linear_position = get_diagonal_indices(self.block_size)
        self.pos2D_mapping = generate_reverse_mapping(self.linear_position)

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        # max_seq_length = find_multiple(max_seq_length, 8)
        # print(max_seq_length)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        # Modified to support parallel prediction positions
        self.register_buffer('position_ids', torch.arange(max_seq_length))

        """row-flatten mask"""
        # causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        # self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)

        """Diagonal flatten mask"""
        self.causal_mask = create_diagonal_causal_mask(self.max_seq_length, self.max_batch_size, self.block_size)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        """2D"""
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head,
                                                 self.config.rope_base, self.cls_token_num, mode="diagonal")
        """1D"""
        # self.freqs_cis = precompute_freqs_cis(grid_size**2, self.config.dim // self.config.n_head,
        #                                          self.config.rope_base, self.cls_token_num)

    def forward(self, idx_embedding: torch.Tensor, cond_idx: torch.Tensor,
                input_pos: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                valid: Optional[torch.Tensor] = None):

        if idx_embedding is not None and cond_idx is not None:
            # Training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, :self.cls_token_num]
            token_embeddings = idx_embedding
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
            """WARNING!!! the PE should especially fit"""
            freqs_cis = self.freqs_cis[1:]

            """Train time mask"""  # TODO: try stochastic mask
            bs = token_embeddings.shape[0]
            print("Prefix: ", h.shape[1] - self.block_size, "image_latent", self.block_size)
            self.causal_mask = create_diagonal_causal_mask(h.shape[1], bs, block_size=self.block_size, verbose=False)
            mask = self.causal_mask[:bs, None, :]

            """Progressive Mask"""
            if mask is not None:
                mask = mask.to(h.device)

        else:
            # Parallel prediction at multiple positions
            if cond_idx is not None:
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, :self.cls_token_num]
            else:
                token_embeddings = idx_embedding

            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
            # Get the frequencies for the specific positions we want to predict
            # print("freq_shape is ", self.freqs_cis.shape, "freq is taking: ", input_pos)
            freqs_cis = self.freqs_cis[input_pos]

            # TODO:
            # freqs_cis_left, freqs_cis_right = batch_get_neighbors_freq_cis(self.freqs_cis, input_pos,
            #                                                                self.pos2D_mapping)
            # freqs_cis = (freqs_cis_left, freqs_cis_right)
            # print("freq_shape is ", freqs_cis.shape, "freq is taking: ", input_pos)

            # Create attention mask for parallel prediction
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            # print("Show mask: ", mask[0], "mask shape: ", print(mask.shape))

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, input_pos, mask)

        h = self.norm(h)
        logits = self.output(h).float()

        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # Calculate loss if needed
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, mode="diagonal"):
    """
    Precompute 2D rotary embeddings, now supporting parallel position prediction.

    Args:
        grid_size: Size of the 2D grid
        n_elem: Number of elements (usually head_dim)
        base: Base for frequency computation
        cls_token_num: Number of class tokens
    Returns:
        Precomputed frequency tensor
    """

    assert mode in ["diagonal", "row-flatten"]

    # Split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 4)

    # Create 2D grid frequencies
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)

    # Convert to complex components
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)

    if mode == "diagonal":
        # Create a tensor to store the diagonally flattened frequencies
        cache = torch.zeros(grid_size ** 2, n_elem // 2, 2, device=cache_grid.device)
        # Get diagonal indices
        diag_positions = get_diagonal_indices(grid_size)
        # Fill the diagonally flattened cache based on diagonal indices

        """diagonal flatten!!!"""

        for diag in diag_positions:
            for (r, c, seq_pos) in diag:
                cache[seq_pos] = cache_grid[r, c]

    elif mode == "row-flatten":
        cache = cache_grid.flatten(0, 1)  # (grid_size**2, head_dim // 2, 2)

    # Add class token positions
    cond_cache = torch.cat([
        torch.zeros(cls_token_num, n_elem // 2, 2),
        cache
    ])  # (cls_token_num + grid_size**2, head_dim // 2, 2)

    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to input tensors, supporting parallel position prediction.

    Args:
        x: Input tensor of shape (bs, seq_len, n_head, head_dim)
        freqs_cis: Frequency tensor of shape (seq_len, head_dim//2, 2) or (num_positions, head_dim//2, 2)
    Returns:
        Tensor with rotary embeddings applied
    """
    # Reshape input tensor into pairs of features
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.to(x.device)
    # Check if we're in parallel prediction mode (multiple positions)
    if len(freqs_cis.shape) == 3 and freqs_cis.size(0) != x.size(1):
        # For parallel prediction, we need to adjust freqs_cis shape
        # freqs_cis shape should be (num_positions, head_dim//2, 2)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, num_positions, 1, head_dim//2, 2)
    else:
        # Original shape transformation for standard prediction
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # Apply rotary embeddings using complex number multiplication
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)

    # Reshape back to original dimensions
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def apply_mixed_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, pos1: tuple, pos2: tuple):
    """
    Apply rotary embeddings separately for two parts of an embedding based on positions.

    Args:
        x: Input tensor of shape (bs, seq_len, n_head, head_dim)
        freqs_cis: Precomputed frequency tensor from `precompute_freqs_cis_2d`
        pos1: Tuple (row, col) for the first position
        pos2: Tuple (row, col) for the second position
    Returns:
        Tensor with rotary embeddings applied to the first and second halves separately.
    """
    head_dim = x.size(-1)
    half_dim = head_dim // 2

    # Split input into first and second halves
    x_first_half, x_second_half = x[..., :half_dim], x[..., half_dim:]

    # Extract frequency embeddings for pos1 and pos2 directly
    freqs_pos1 = freqs_cis[pos1]  # (head_dim//2, 2)
    freqs_pos2 = freqs_cis[pos2]  # (head_dim//2, 2)

    # Apply rotary embeddings for the first and second halves
    x_first_half = apply_rotary_emb(x_first_half, freqs_pos1)
    x_second_half = apply_rotary_emb(x_second_half, freqs_pos2)

    # Concatenate the modified halves
    x_out = torch.cat([x_first_half, x_second_half], dim=-1)
    return x_out


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs))  # 6.6B


def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs))  # 3.1B


def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs))  # 1.2B


### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))  # 3.9B


def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))  # 1.4B


def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))  # 775M


def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))  # 343M


def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))  # 111M


GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B,
}
