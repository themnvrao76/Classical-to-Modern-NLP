import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[: x.size(1)].to(dtype=x.dtype, device=x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, key_padding_mask=None, attention_mask=None):
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[None, None, :, :], torch.finfo(scores.dtype).min)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(query.size(0), query.size(1), -1)
        return self.out_proj(context), weights


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        attended, _ = self.self_attention(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, target_padding_mask=None, memory_padding_mask=None, causal_mask=None):
        attended, _ = self.self_attention(
            x, x, x, key_padding_mask=target_padding_mask, attention_mask=causal_mask
        )
        x = self.norm1(x + self.dropout(attended))
        attended, cross_weights = self.cross_attention(
            x, memory, memory, key_padding_mask=memory_padding_mask
        )
        x = self.norm2(x + self.dropout(attended))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x, cross_weights


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, dropout=0.1, max_length=5000, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_length)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, tgt_vocab_size)

    def _embed(self, tokens, embedding):
        return self.embedding_dropout(self.position(embedding(tokens) * math.sqrt(self.d_model)))

    def encode(self, source):
        source_mask = source.ne(self.pad_id)
        memory = self._embed(source, self.src_embedding)
        for layer in self.encoder_layers:
            memory = layer(memory, source_mask)
        return memory, source_mask

    def decode(self, target, memory, source_mask):
        target_mask = target.ne(self.pad_id)
        length = target.size(1)
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=target.device).tril()
        hidden = self._embed(target, self.tgt_embedding)
        cross_weights = None
        for layer in self.decoder_layers:
            hidden, cross_weights = layer(hidden, memory, target_mask, source_mask, causal_mask)
        return self.output(hidden), cross_weights

    def forward(self, source, target_inputs):
        memory, source_mask = self.encode(source)
        return self.decode(target_inputs, memory, source_mask)

    @torch.no_grad()
    def translate(self, source, bos_id, max_length=32, eos_id=None):
        memory, source_mask = self.encode(source)
        generated = torch.full((source.size(0), 1), bos_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(max_length):
            logits, _ = self.decode(generated, memory, source_mask)
            next_token = logits[:, -1].argmax(dim=-1)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if eos_id is not None:
                finished |= next_token.eq(eos_id)
                if finished.all():
                    break
        return generated[:, 1:]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = Transformer(10_000, 10_000)
    source = torch.randint(1, 10_000, (2, 12))
    target = torch.randint(1, 10_000, (2, 9))
    logits, cross_attention = model(source, target)
    print(model)
    print(f"parameters: {parameter_count(model):,}")
    print("logits:", tuple(logits.shape))
    print("cross attention:", tuple(cross_attention.shape))
