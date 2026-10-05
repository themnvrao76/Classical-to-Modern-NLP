import torch
import torch.nn as nn


class BART(nn.Module):
    def __init__(self, vocab_size=50265, d_model=768, encoder_layers=6,
                 decoder_layers=6, heads=12, d_ff=3072, max_position=1024):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.position = nn.Embedding(max_position, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, heads, d_ff, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, heads, d_ff, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight

    def embed(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        return self.token(ids) + self.position(pos)[None]

    def forward(self, input_ids, decoder_input_ids, attention_mask=None):
        src = self.embed(input_ids)
        src_key_padding = None if attention_mask is None else ~attention_mask.bool()
        memory = self.encoder(src, src_key_padding_mask=src_key_padding)
        tgt = self.embed(decoder_input_ids)
        seq = tgt.shape[1]
        causal = torch.full((seq, seq), float("-inf"), device=tgt.device)
        causal = torch.triu(causal, diagonal=1)
        out = self.decoder(
            tgt, memory,
            tgt_mask=causal,
            memory_key_padding_mask=src_key_padding,
        )
        return self.lm_head(self.norm(out))


if __name__ == "__main__":
    model = BART(vocab_size=10000, d_model=256, encoder_layers=4, decoder_layers=4, heads=8, d_ff=1024)
    src = torch.randint(2, 10000, (2, 24))
    tgt = torch.randint(2, 10000, (2, 16))
    logits = model(src, tgt)
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
