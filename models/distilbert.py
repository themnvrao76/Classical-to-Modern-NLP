import torch
import torch.nn as nn


class DistilBERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, layers=6, heads=12,
                 intermediate=3072, max_position=512, num_classes=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(max_position, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(
            hidden_size, heads, intermediate,
            dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token(input_ids) + self.position(pos)[None]
        x = self.dropout(self.embedding_norm(x))
        key_padding = None if attention_mask is None else ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding)
        pooled = self.norm(x[:, 0])
        return self.classifier(pooled)


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    return torch.nn.functional.kl_div(student, teacher, reduction="batchmean") * temperature ** 2


if __name__ == "__main__":
    model = DistilBERT(hidden_size=256, layers=4, heads=8, intermediate=1024)
    ids = torch.randint(0, 30522, (2, 32))
    logits = model(ids)
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
