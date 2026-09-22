import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_gates = nn.Linear(input_size, 2 * hidden_size)
        self.h_gates = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.x_candidate = nn.Linear(input_size, hidden_size)
        self.h_candidate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x_reset, x_update = self.x_gates(x).chunk(2, dim=-1)
        h_reset, h_update = self.h_gates(hidden).chunk(2, dim=-1)
        reset = torch.sigmoid(x_reset + h_reset)
        update = torch.sigmoid(x_update + h_update)
        candidate = torch.tanh(self.x_candidate(x) + reset * self.h_candidate(hidden))
        return (1.0 - update) * candidate + update * hidden


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        if x.ndim != 3:
            raise ValueError("expected input with shape (batch, sequence, features)")
        batch_size = x.size(0)
        if hidden is None:
            hidden = x.new_zeros(batch_size, self.hidden_size)
        outputs = []
        for step in x.unbind(dim=1):
            hidden = self.cell(step, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), hidden


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = GRU(embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor | None = None):
        sequence, hidden = self.gru(self.embedding(tokens), hidden)
        return self.output(sequence), hidden


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = GRULanguageModel(vocab_size=10_000)
    tokens = torch.randint(0, 10_000, (4, 32))
    logits, hidden = model(tokens)
    print("logits:", tuple(logits.shape))
    print("hidden:", tuple(hidden.shape))
    print("parameters:", f"{count_parameters(model):,}")
