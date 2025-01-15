from torch import nn
import torch.nn.functional as F
import torch
from typing import Optional


class ModelArgs:
    vocab_size: int = 16384
    metalearner_config: Optional[dict] = None
    metalearner_hidden_dim: int = 4096
    metalearner_activation: str = 'gelu'
    gpt_dim: int = 768

    def __post_init__(self):

        # DEFAULT
        if self.metalearner_config is None:
            self.metalearner_config = {
                'input_dim': self.vocab_size * 2,  # 左侧和上方context的logits维度之和
                'hidden_dim': self.metalearner_hidden_dim,
                'output_dim': self.vocab_size,  # 输出维度等于词汇表大小
                'activation': self.metalearner_activation,
            }


class MetaLearner(nn.Module):
    def __init__(self, config: ModelArgs, gpt_dim):
        super().__init__()
        # half of GPT's hidden_dim
        self.token_embedding = nn.Embedding(config.vocab_size + 3, gpt_dim//2)  # added <up +2/left +1 bonds>
        # self.out = nn.Linear(gpt_dim, gpt_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, left_idx, up_idx):
        left_emb = self.token_embedding(left_idx)
        up_emb = self.token_embedding(up_idx)
        """Method I: Concat"""
        embeddings = torch.cat((left_emb, up_emb), dim=-1)
        """Method II: Addition"""
        # embeddings = left_emb + up_emb
        # if train and use_dropout:
        #     embeddings = self.token_drop(embeddings)
        # embeddings = self.out(embeddings)
        return embeddings


if __name__ == '__main__':
    model_args = ModelArgs()
    meta_mlp = MetaLearner(model_args, gpt_dim=model_args.gpt_dim)
    # left = torch.randn(1, model_args.vocab_size)
    # right = torch.randn(1, model_args.vocab_size)
    left = torch.randint(1, 16384, (3,))
    up = torch.randint(1, 16384, (3,))
    # print(left.shape, right.shape)
    out = meta_mlp(left, up)
    print(out.shape)
