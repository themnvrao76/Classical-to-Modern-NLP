import torch
import torch.nn as nn


class T5Block(nn.Module):
    def __init__(self, d_model=512, heads=8, d_ff=2048, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        if decoder:
            self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
            self.norm_cross = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, memory=None, causal_mask=None, key_padding_mask=None):
        h = self.norm1(x)
        attn, _ = self.self_attn(
            h, h, h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn
        if self.decoder and memory is not None:
            h = self.norm_cross(x)
            cross, _ = self.cross_attn(h, memory, memory, need_weights=False)
            x = x + cross
        x = x + self.ff(self.norm2(x))
        return x


class T5(nn.Module):
    def __init__(self, vocab_size=32128, d_model=512, layers=6, heads=8, d_ff=2048):
        super().__init__()
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.ModuleList(T5Block(d_model, heads, d_ff, decoder=False) for _ in range(layers))
        self.decoder = nn.ModuleList(T5Block(d_model, heads, d_ff, decoder=True) for _ in range(layers))
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight

    def encode(self, input_ids, attention_mask=None):
        x = self.shared(input_ids)
        key_padding = None if attention_mask is None else ~attention_mask.bool()
        for layer in self.encoder:
            x = layer(x, key_padding_mask=key_padding)
        return self.encoder_norm(x)

    def forward(self, input_ids, decoder_input_ids, attention_mask=None):
        memory = self.encode(input_ids, attention_mask)
        y = self.shared(decoder_input_ids)
        seq = y.shape[1]
        causal = torch.full((seq, seq), float("-inf"), device=y.device)
        causal = torch.triu(causal, diagonal=1)
        for layer in self.decoder:
            y = layer(y, memory=memory, causal_mask=causal)
        return self.lm_head(self.decoder_norm(y))


if __name__ == "__main__":
    model = T5(vocab_size=10000, d_model=256, layers=4, heads=8, d_ff=1024)
    src = torch.randint(0, 10000, (2, 24))
    tgt = torch.randint(0, 10000, (2, 16))
    logits = model(src, tgt)
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
