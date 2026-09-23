import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.input_proj(x) + self.hidden_proj(h)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            LSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])

    def initial_state(self, batch_size, device, dtype):
        return [
            (
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            )
            for _ in range(self.num_layers)
        ]

    def step(self, x, states):
        next_states = []
        for cell, state in zip(self.cells, states):
            x, c = cell(x, state)
            next_states.append((x, c))
        return x, next_states


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = StackedLSTM(embedding_dim, hidden_size, num_layers)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        states = self.rnn.initial_state(tokens.size(0), tokens.device, embedded.dtype)
        outputs = []
        for t in range(tokens.size(1)):
            hidden, states = self.rnn.step(embedded[:, t], states)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), states


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        scores = self.energy(
            torch.tanh(self.encoder_proj(encoder_outputs) + self.decoder_proj(decoder_hidden).unsqueeze(1))
        ).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, attention_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = AdditiveAttention(hidden_size, hidden_size, attention_dim)
        self.rnn = StackedLSTM(embedding_dim + hidden_size, hidden_size, num_layers)
        self.output = nn.Linear(hidden_size * 2, vocab_size)

    def step(self, token, encoder_outputs, states, mask=None):
        context, weights = self.attention(encoder_outputs, states[-1][0], mask)
        rnn_input = torch.cat((self.embedding(token), context), dim=-1)
        hidden, states = self.rnn.step(rnn_input, states)
        logits = self.output(torch.cat((hidden, context), dim=-1))
        return logits, states, weights

    def forward(self, tokens, encoder_outputs, states, mask=None):
        logits, alignments = [], []
        for t in range(tokens.size(1)):
            step_logits, states, weights = self.step(tokens[:, t], encoder_outputs, states, mask)
            logits.append(step_logits)
            alignments.append(weights)
        return torch.stack(logits, dim=1), states, torch.stack(alignments, dim=1)


class BahdanauSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, hidden_size=512,
                 attention_dim=512, num_layers=2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_size, num_layers)
        self.decoder = AttentionDecoder(tgt_vocab_size, embedding_dim, hidden_size, attention_dim, num_layers)

    def forward(self, source, target_inputs):
        encoder_outputs, states = self.encoder(source)
        source_mask = source.ne(self.pad_id)
        logits, _, alignments = self.decoder(target_inputs, encoder_outputs, states, source_mask)
        return logits, alignments

    @torch.no_grad()
    def translate(self, source, bos_id, max_length=32, eos_id=None):
        encoder_outputs, states = self.encoder(source)
        mask = source.ne(self.pad_id)
        token = torch.full((source.size(0),), bos_id, dtype=torch.long, device=source.device)
        generated, alignments = [], []
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(max_length):
            logits, states, weights = self.decoder.step(token, encoder_outputs, states, mask)
            token = logits.argmax(dim=-1)
            generated.append(token)
            alignments.append(weights)
            if eos_id is not None:
                finished |= token.eq(eos_id)
                if finished.all():
                    break
        return torch.stack(generated, dim=1), torch.stack(alignments, dim=1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = BahdanauSeq2Seq(10_000, 10_000)
    source = torch.randint(1, 10_000, (2, 12))
    target = torch.randint(1, 10_000, (2, 9))
    logits, alignments = model(source, target)
    print(model)
    print(f"parameters: {parameter_count(model):,}")
    print("logits:", tuple(logits.shape))
    print("alignments:", tuple(alignments.shape))
