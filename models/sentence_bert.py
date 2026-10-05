import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceBERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=384, layers=6, heads=6,
                 intermediate=1536, max_position=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_position, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size, heads, intermediate,
            dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token(input_ids) + self.position(pos)[None]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)


def cosine_similarity_matrix(a, b):
    return a @ b.t()


if __name__ == "__main__":
    model = SentenceBERT(vocab_size=10000, hidden_size=256, layers=4, heads=8, intermediate=1024)
    ids = torch.randint(1, 10000, (4, 32))
    embeddings = model(ids)
    print("embeddings:", tuple(embeddings.shape))
    print("similarity:", tuple(cosine_similarity_matrix(embeddings, embeddings).shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
