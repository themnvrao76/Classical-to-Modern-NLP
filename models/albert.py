import torch
import torch.nn as nn


class AlbertLayer(nn.Module):
    def __init__(self, hidden_size, heads, intermediate):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_size),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))


class ALBERT(nn.Module):
    def __init__(self, vocab_size=30000, embedding_size=128, hidden_size=768,
                 layers=12, heads=12, intermediate=3072, max_position=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_size)
        self.position = nn.Embedding(max_position, embedding_size)
        self.embedding_norm = nn.LayerNorm(embedding_size)
        self.project = nn.Linear(embedding_size, hidden_size)
        self.shared_layer = AlbertLayer(hidden_size, heads, intermediate)
        self.layers = layers
        self.mlm_transform = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.GELU(),
            nn.LayerNorm(embedding_size),
        )
        self.decoder = nn.Linear(embedding_size, vocab_size, bias=False)
        self.decoder.weight = self.token.weight

    def forward(self, input_ids, attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token(input_ids) + self.position(positions)[None]
        x = self.project(self.embedding_norm(x))
        key_padding = None if attention_mask is None else ~attention_mask.bool()
        for _ in range(self.layers):
            x = self.shared_layer(x, key_padding)
        return self.decoder(self.mlm_transform(x))


if __name__ == "__main__":
    model = ALBERT(hidden_size=256, layers=6, heads=8, intermediate=1024)
    ids = torch.randint(0, 30000, (2, 32))
    logits = model(ids)
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
