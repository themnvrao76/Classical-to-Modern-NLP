import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, heads=8, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.r = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, rel_pos):
        b, q_len, d = query.shape
        k_len = key_value.shape[1]
        q = self.q(query).view(b, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).view(b, k_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(key_value).view(b, k_len, self.heads, self.head_dim).transpose(1, 2)
        r = self.r(rel_pos).view(k_len, self.heads, self.head_dim).permute(1, 0, 2)
        content = torch.einsum("bhqd,bhkd->bhqk", q, k)
        relative = torch.einsum("bhqd,hkd->bhqk", q, r)
        scores = (content + relative) * self.scale
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).triu(
            diagonal=k_len - q_len + 1
        )
        scores = scores.masked_fill(causal[None, None], float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(1, 2).reshape(b, q_len, d)
        return self.out(out)


class TransformerXLLayer(nn.Module):
    def __init__(self, d_model=512, heads=8, d_ff=2048):
        super().__init__()
        self.attn = RelativeMultiHeadAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, memory, rel_pos):
        kv = torch.cat([memory, x], dim=1) if memory is not None else x
        x = x + self.attn(self.norm1(x), self.norm1(kv), rel_pos)
        return x + self.ff(self.norm2(x))


class TransformerXL(nn.Module):
    def __init__(self, vocab_size=30000, d_model=512, layers=6, heads=8,
                 d_ff=2048, memory_len=128, max_relative=1024):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.rel = nn.Embedding(max_relative, d_model)
        self.layers = nn.ModuleList(TransformerXLLayer(d_model, heads, d_ff) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.memory_len = memory_len
        self.max_relative = max_relative

    def forward(self, input_ids, memories=None):
        x = self.token(input_ids)
        memories = memories or [None] * len(self.layers)
        new_memories = []
        for i, layer in enumerate(self.layers):
            mem = memories[i]
            k_len = x.shape[1] + (0 if mem is None else mem.shape[1])
            positions = torch.arange(k_len - 1, -1, -1, device=x.device).clamp_max(self.max_relative - 1)
            rel_pos = self.rel(positions)
            old = x
            x = layer(x, mem, rel_pos)
            merged = torch.cat([mem, old], dim=1) if mem is not None else old
            new_memories.append(merged[:, -self.memory_len:].detach())
        return self.head(self.norm(x)), new_memories


if __name__ == "__main__":
    model = TransformerXL(vocab_size=10000, d_model=256, layers=4, heads=8, d_ff=1024)
    ids = torch.randint(0, 10000, (2, 32))
    logits, memory = model(ids)
    print("logits:", tuple(logits.shape), "memory:", tuple(memory[0].shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
