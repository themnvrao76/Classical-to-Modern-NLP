import torch
from torch import nn


class ElmanRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.input_projection(x) + self.hidden_projection(hidden))


class ElmanRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ElmanRNNCell(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("expected input shaped [batch, sequence, features]")

        batch_size = x.size(0)
        if hidden is None:
            hidden = x.new_zeros(batch_size, self.hidden_size)
        elif hidden.shape != (batch_size, self.hidden_size):
            raise ValueError("hidden state has an incompatible shape")

        states = []
        for step in x.unbind(dim=1):
            hidden = self.cell(step, hidden)
            states.append(hidden)

        if not states:
            return x.new_empty(batch_size, 0, self.hidden_size), hidden
        return torch.stack(states, dim=1), hidden


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = ElmanRNN(embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("expected token ids shaped [batch, sequence]")
        states, hidden = self.rnn(self.embedding(tokens), hidden)
        return self.output(states), hidden


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(7)
    model = RNNLanguageModel(vocab_size=10_000)
    tokens = torch.randint(0, 10_000, (4, 32))
    logits, hidden = model(tokens)

    targets = torch.randint(0, 10_000, (4, 32))
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()

    print(f"logits: {tuple(logits.shape)}")
    print(f"final hidden: {tuple(hidden.shape)}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"loss: {loss.item():.4f}")
