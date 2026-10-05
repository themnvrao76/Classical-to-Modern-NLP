import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base + update


class LoRAAttention(nn.Module):
    def __init__(self, d_model=512, heads=8, rank=8):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.q = LoRALinear(d_model, d_model, rank=rank, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = LoRALinear(d_model, d_model, rank=rank, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(b, n, d))


def trainable_parameter_summary(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    module = LoRAAttention(d_model=256, heads=8, rank=8)
    x = torch.randn(2, 32, 256)
    y = module(x)
    total, trainable = trainable_parameter_summary(module)
    print("output:", tuple(y.shape))
    print("total:", total, "trainable:", trainable)
