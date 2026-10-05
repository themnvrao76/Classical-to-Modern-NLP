import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k):
    seq = q.shape[-2]
    dim = q.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    positions = torch.arange(seq, device=q.device).float()
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    cos = emb.cos()[None, None]
    sin = emb.sin()[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=512, query_heads=8, kv_heads=2):
        super().__init__()
        assert d_model % query_heads == 0 and query_heads % kv_heads == 0
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // query_heads
        self.q = nn.Linear(d_model, query_heads * self.head_dim, bias=False)
        self.k = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.query_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k)
        if cache is not None:
            old_k, old_v = cache
            k = torch.cat([old_k, k], dim=-2)
            v = torch.cat([old_v, v], dim=-2)
        repeat = self.query_heads // self.kv_heads
        k_attn = k.repeat_interleave(repeat, dim=1)
        v_attn = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=cache is None)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.o(out), (k.detach(), v.detach())


class ModernDecoderBlock(nn.Module):
    def __init__(self, d_model=512, query_heads=8, kv_heads=2, d_ff=1536):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, query_heads, kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, cache=None):
        a, cache = self.attn(self.norm1(x), cache)
        x = x + a
        h = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x, cache


if __name__ == "__main__":
    block = ModernDecoderBlock(d_model=256, query_heads=8, kv_heads=2, d_ff=768)
    x = torch.randn(2, 16, 256)
    y, cache = block(x)
    print("output:", tuple(y.shape), "key cache:", tuple(cache[0].shape))
    print("parameters:", sum(p.numel() for p in block.parameters()))
