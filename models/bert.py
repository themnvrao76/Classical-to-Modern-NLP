import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.out(context)


class BERTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attention_norm(x + self.dropout(self.attention(x, attention_mask)))
        return self.output_norm(x + self.ffn(x))


class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [BERTBlock(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)]
        )
        self.mlm_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.nsp = nn.Linear(hidden_size, 2)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

    def encode(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError("sequence exceeds maximum position embeddings")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embeddings(input_ids)
        x = x + self.position_embeddings(positions) + self.segment_embeddings(token_type_ids)
        x = self.embedding_dropout(self.embedding_norm(x))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(input_ids, token_type_ids, attention_mask)
        mlm_hidden = self.mlm_transform(hidden)
        mlm_logits = F.linear(mlm_hidden, self.token_embeddings.weight, self.mlm_bias)
        nsp_logits = self.nsp(hidden[:, 0])
        return mlm_logits, nsp_logits

    @staticmethod
    def pretraining_loss(
        mlm_logits: torch.Tensor,
        mlm_labels: torch.Tensor,
        nsp_logits: torch.Tensor,
        nsp_labels: torch.Tensor,
    ) -> torch.Tensor:
        mlm_loss = F.cross_entropy(
            mlm_logits.reshape(-1, mlm_logits.size(-1)),
            mlm_labels.reshape(-1),
            ignore_index=-100,
        )
        nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
        return mlm_loss + nsp_loss


if __name__ == "__main__":
    model = BERT()
    input_ids = torch.randint(1, 30522, (2, 32))
    input_ids[:, 0] = 101
    mlm_logits, nsp_logits = model(input_ids)
    print("MLM logits:", tuple(mlm_logits.shape))
    print("NSP logits:", tuple(nsp_logits.shape))
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
