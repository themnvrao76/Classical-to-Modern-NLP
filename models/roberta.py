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
            scores = scores.masked_fill(
                ~attention_mask[:, None, None, :].bool(),
                torch.finfo(scores.dtype).min,
            )
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.out(context)


class RoBERTaBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attention_norm(x + self.attention_dropout(self.attention(x, attention_mask)))
        y = self.output(F.gelu(self.intermediate(x)))
        return self.output_norm(x + self.output_dropout(y))


class RoBERTa(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50265,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 514,
        pad_token_id: int = 1,
        mask_token_id: int = 50264,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.max_position_embeddings = max_position_embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size, padding_idx=pad_token_id)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [RoBERTaBlock(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)]
        )
        self.lm_dense = nn.Linear(hidden_size, hidden_size)
        self.lm_norm = nn.LayerNorm(hidden_size)
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))

    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.pad_token_id).long()
        incremental = torch.cumsum(mask, dim=1) * mask
        return incremental + self.pad_token_id

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.size(1) + self.pad_token_id >= self.max_position_embeddings:
            raise ValueError("sequence exceeds maximum position embeddings")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        positions = self.create_position_ids(input_ids)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embedding_dropout(self.embedding_norm(x))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        hidden = self.encode(input_ids, attention_mask)
        lm_hidden = self.lm_norm(F.gelu(self.lm_dense(hidden)))
        logits = F.linear(lm_hidden, self.token_embeddings.weight, self.lm_bias)
        if labels is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    def dynamic_mask(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
        mlm_probability: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = input_ids.clone()
        labels = input_ids.clone()
        probability = torch.full(input_ids.shape, mlm_probability, device=input_ids.device)
        probability.masked_fill_(input_ids.eq(self.pad_token_id), 0.0)
        if special_tokens_mask is not None:
            probability.masked_fill_(special_tokens_mask.bool(), 0.0)
        selected = torch.bernoulli(probability).bool()
        labels[~selected] = -100

        replace_with_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool() & selected
        masked[replace_with_mask] = self.mask_token_id

        remaining = selected & ~replace_with_mask
        replace_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool() & remaining
        random_words = torch.randint(self.vocab_size, input_ids.shape, device=input_ids.device)
        masked[replace_random] = random_words[replace_random]
        return masked, labels


if __name__ == "__main__":
    model = RoBERTa()
    tokens = torch.randint(4, 50264, (2, 32))
    tokens[:, 0] = 0
    masked, labels = model.dynamic_mask(tokens, special_tokens_mask=tokens.eq(0))
    logits, loss = model(masked, labels=labels)
    print("logits:", tuple(logits.shape))
    print("loss:", float(loss))
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
