import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
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
        for t in range(tokens.size(1)):
            _, states = self.rnn.step(embedded[:, t], states)
        return states


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = StackedLSTM(embedding_dim, hidden_size, num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, states):
        embedded = self.embedding(tokens)
        logits = []
        for t in range(tokens.size(1)):
            hidden, states = self.rnn.step(embedded[:, t], states)
            logits.append(self.output(hidden))
        return torch.stack(logits, dim=1), states

    @torch.no_grad()
    def generate(self, start_tokens, states, max_length, eos_id=None):
        token = start_tokens
        generated = []
        finished = torch.zeros(token.size(0), dtype=torch.bool, device=token.device)
        for _ in range(max_length):
            embedded = self.embedding(token)
            hidden, states = self.rnn.step(embedded, states)
            token = self.output(hidden).argmax(dim=-1)
            generated.append(token)
            if eos_id is not None:
                finished |= token.eq(eos_id)
                if finished.all():
                    break
        return torch.stack(generated, dim=1), states


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_size, num_layers)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_size, num_layers)

    def forward(self, source, target_inputs):
        encoder_state = self.encoder(source)
        logits, _ = self.decoder(target_inputs, encoder_state)
        return logits

    @torch.no_grad()
    def translate(self, source, bos_id, max_length=32, eos_id=None):
        states = self.encoder(source)
        start = torch.full((source.size(0),), bos_id, dtype=torch.long, device=source.device)
        tokens, _ = self.decoder.generate(start, states, max_length, eos_id)
        return tokens


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = Seq2Seq(10_000, 10_000)
    source = torch.randint(0, 10_000, (2, 12))
    target = torch.randint(0, 10_000, (2, 9))
    logits = model(source, target)
    print(model)
    print(f"parameters: {parameter_count(model):,}")
    print("logits:", tuple(logits.shape))
