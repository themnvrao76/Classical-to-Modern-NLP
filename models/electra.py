import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, layers, heads, intermediate, max_position=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(max_position, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size, heads, intermediate,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token(input_ids) + self.position(pos)[None]
        key_padding = None if attention_mask is None else ~attention_mask.bool()
        return self.norm(self.encoder(x, src_key_padding_mask=key_padding))


class ELECTRA(nn.Module):
    def __init__(self, vocab_size=30522, generator_hidden=256, discriminator_hidden=768,
                 generator_layers=4, discriminator_layers=12):
        super().__init__()
        self.generator = Encoder(
            vocab_size, generator_hidden, generator_layers, 4, generator_hidden * 4
        )
        self.generator_head = nn.Linear(generator_hidden, vocab_size)
        self.discriminator = Encoder(
            vocab_size, discriminator_hidden, discriminator_layers, 12, discriminator_hidden * 4
        )
        self.discriminator_head = nn.Linear(discriminator_hidden, 1)

    def generator_logits(self, masked_ids, attention_mask=None):
        return self.generator_head(self.generator(masked_ids, attention_mask))

    def discriminator_logits(self, corrupted_ids, attention_mask=None):
        hidden = self.discriminator(corrupted_ids, attention_mask)
        return self.discriminator_head(hidden).squeeze(-1)

    def forward(self, masked_ids, corrupted_ids, attention_mask=None):
        return {
            "generator_logits": self.generator_logits(masked_ids, attention_mask),
            "discriminator_logits": self.discriminator_logits(corrupted_ids, attention_mask),
        }


if __name__ == "__main__":
    model = ELECTRA(
        vocab_size=10000,
        generator_hidden=128,
        discriminator_hidden=256,
        generator_layers=2,
        discriminator_layers=4,
    )
    ids = torch.randint(0, 10000, (2, 24))
    out = model(ids, ids)
    print("generator:", tuple(out["generator_logits"].shape))
    print("discriminator:", tuple(out["discriminator_logits"].shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
