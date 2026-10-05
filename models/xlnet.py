import torch
import torch.nn as nn
import torch.nn.functional as F


class XLNetAttention(nn.Module):
    def __init__(self, d_model=512, heads=8, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_stream, content, permutation_mask=None):
        b, q_len, d = q_stream.shape
        k_len = content.shape[1]
        q = self.q(q_stream).view(b, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(content).view(b, k_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(content).view(b, k_len, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if permutation_mask is not None:
            scores = scores.masked_fill(permutation_mask[:, None].bool(), float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v)
        return self.o(out.transpose(1, 2).reshape(b, q_len, d))


class XLNetLayer(nn.Module):
    def __init__(self, d_model=512, heads=8, d_ff=2048):
        super().__init__()
        self.attn = XLNetAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, content, query, permutation_mask):
        content = content + self.attn(self.norm1(content), self.norm1(content), permutation_mask)
        query = query + self.attn(self.norm1(query), self.norm1(content), permutation_mask)
        content = content + self.ff(self.norm2(content))
        query = query + self.ff(self.norm2(query))
        return content, query


class XLNet(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, layers=6, heads=8,
                 d_ff=2048, max_position=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_position, d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList(XLNetLayer(d_model, heads, d_ff) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids, permutation_mask=None):
        seq = input_ids.shape[1]
        positions = torch.arange(seq, device=input_ids.device)
        content = self.token(input_ids) + self.position(positions)[None]
        query = self.mask_embedding.expand(input_ids.shape[0], seq, -1) + self.position(positions)[None]
        for layer in self.layers:
            content, query = layer(content, query, permutation_mask)
        return self.head(self.norm(query))


if __name__ == "__main__":
    model = XLNet(vocab_size=10000, d_model=256, layers=4, heads=8, d_ff=1024)
    ids = torch.randint(0, 10000, (2, 24))
    order = torch.rand(2, 24).argsort(dim=1)
    rank = order.argsort(dim=1)
    mask = rank[:, :, None] < rank[:, None, :]
    logits = model(ids, mask)
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
