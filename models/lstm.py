import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.hidden_projection = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        gates = self.input_projection(x) + self.hidden_projection(hidden)
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)

        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError("expected input shaped [batch, sequence, features]")

        batch_size = x.size(0)
        if state is None:
            hidden = x.new_zeros(batch_size, self.hidden_size)
            cell = x.new_zeros(batch_size, self.hidden_size)
        else:
            hidden, cell = state
            expected = (batch_size, self.hidden_size)
            if hidden.shape != expected or cell.shape != expected:
                raise ValueError("hidden or cell state has an incompatible shape")

        outputs = []
        for step in x.unbind(dim=1):
            hidden, cell = self.cell(step, (hidden, cell))
            outputs.append(hidden)

        if not outputs:
            sequence = x.new_empty(batch_size, 0, self.hidden_size)
        else:
            sequence = torch.stack(outputs, dim=1)
        return sequence, (hidden, cell)


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if tokens.ndim != 2:
            raise ValueError("expected token ids shaped [batch, sequence]")
        sequence, state = self.lstm(self.embedding(tokens), state)
        return self.output(sequence), state


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(7)
    model = LSTMLanguageModel(vocab_size=10_000)
    tokens = torch.randint(0, 10_000, (4, 32))
    logits, (hidden, cell) = model(tokens)

    targets = torch.randint(0, 10_000, (4, 32))
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()

    print(f"logits: {tuple(logits.shape)}")
    print(f"final hidden: {tuple(hidden.shape)}")
    print(f"final cell: {tuple(cell.shape)}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"loss: {loss.item():.4f}")
