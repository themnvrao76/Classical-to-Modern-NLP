"""GPT-1: decoder-only Transformer for generative pretraining.

Implements the core architecture from Radford et al. (2018): learned token and
position embeddings, masked multi-head self-attention, position-wise MLPs,
residual connections, layer normalization, and an autoregressive LM head.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.resid_dropout(self.out(context))


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))
        mlp = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return self.ln2(x + self.dropout(mlp))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        max_seq_len: int = 512,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds {self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        )
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.max_seq_len :]
            logits = self(context)[:, -1] / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids


if __name__ == "__main__":
    model = GPT()
    tokens = torch.randint(0, 10000, (2, 32))
    logits = model(tokens)
    params = sum(p.numel() for p in model.parameters())
    print("logits:", tuple(logits.shape))
    print("parameters:", f"{params:,}")
