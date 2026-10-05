import torch
import torch.nn as nn


class RobertaEmbeddings(nn.Module):
    def __init__(self, vocab_size=50265, hidden_size=768, max_position=514, pad_id=1):
        super().__init__()
        self.word = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.position = nn.Embedding(max_position, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.pad_id = pad_id

    def forward(self, input_ids):
        mask = input_ids.ne(self.pad_id).long()
        positions = torch.cumsum(mask, dim=1) * mask + self.pad_id
        x = self.word(input_ids) + self.position(positions)
        return self.dropout(self.norm(x))


class RoBERTa(nn.Module):
    def __init__(self, vocab_size=50265, hidden_size=768, layers=12, heads=12,
                 intermediate=3072, max_position=514):
        super().__init__()
        self.embeddings = RobertaEmbeddings(vocab_size, hidden_size, max_position)
        layer = nn.TransformerEncoderLayer(
            hidden_size, heads, intermediate, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size, bias=False),
        )
        self.lm_head[-1].weight = self.embeddings.word.weight

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        key_padding = None if attention_mask is None else ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding)
        return self.lm_head(self.norm(x))


if __name__ == "__main__":
    model = RoBERTa(hidden_size=256, layers=4, heads=8, intermediate=1024)
    ids = torch.randint(4, 50265, (2, 32))
    logits = model(ids)
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
