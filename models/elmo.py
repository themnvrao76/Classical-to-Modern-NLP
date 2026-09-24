"""ELMo: Deep contextualized word representations (Peters et al., 2018).

A compact, runnable implementation of the architectural ideas behind ELMo:
character-CNN token encoding, highway layers, stacked bidirectional language
models, and a learned task-specific scalar mixture of representation layers.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.proj = nn.Linear(size, 2 * size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        return gate * F.relu(value) + (1.0 - gate) * x


class CharacterCNN(nn.Module):
    """Build context-independent token vectors from characters."""

    def __init__(
        self,
        char_vocab_size: int = 262,
        char_dim: int = 16,
        filters=((1, 32), (2, 32), (3, 64), (4, 128),
                 (5, 256), (6, 512), (7, 1024)),
        output_dim: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(char_dim, channels, kernel_size=width)
            for width, channels in filters
        )
        cnn_dim = sum(channels for _, channels in filters)
        self.projection = nn.Linear(cnn_dim, output_dim)
        self.highways = nn.ModuleList([Highway(output_dim), Highway(output_dim)])

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        batch, tokens, char_count = chars.shape
        x = self.embedding(chars).reshape(batch * tokens, char_count, -1)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convolutions:
            if char_count < conv.kernel_size[0]:
                raise ValueError("max_chars must be at least the largest CNN filter width")
            feature = F.relu(conv(x))
            pooled.append(feature.amax(dim=-1))

        x = self.projection(torch.cat(pooled, dim=-1))
        for highway in self.highways:
            x = highway(x)
        return x.reshape(batch, tokens, -1)


class ELMo(nn.Module):
    """Two-layer bidirectional language model with learned ELMo scalar mixing."""

    def __init__(
        self,
        vocab_size: int = 10_000,
        char_vocab_size: int = 262,
        char_dim: int = 16,
        token_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if 2 * hidden_dim != token_dim:
            raise ValueError("token_dim must equal 2 * hidden_dim for scalar mixing")

        self.token_encoder = CharacterCNN(char_vocab_size, char_dim, output_dim=token_dim)
        self.bilm1 = nn.LSTM(token_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bilm2 = nn.LSTM(token_dim, hidden_dim, batch_first=True, bidirectional=True)

        # One vocabulary projection is shared by the forward and backward LM heads.
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.scalar_weights = nn.Parameter(torch.zeros(3))
        self.gamma = nn.Parameter(torch.ones(1))
        self.hidden_dim = hidden_dim

    def contextual_layers(self, chars: torch.Tensor):
        token = self.token_encoder(chars)
        layer1, _ = self.bilm1(token)
        layer2, _ = self.bilm2(layer1)
        return token, layer1, layer2

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        layers = self.contextual_layers(chars)
        weights = F.softmax(self.scalar_weights, dim=0)
        mixed = sum(weight * layer for weight, layer in zip(weights, layers))
        return self.gamma * mixed

    def language_model_logits(self, chars: torch.Tensor):
        """Return forward next-token and backward previous-token LM logits."""
        _, _, top = self.contextual_layers(chars)
        forward_state = top[..., : self.hidden_dim]
        backward_state = top[..., self.hidden_dim :]
        return self.lm_head(forward_state), self.lm_head(backward_state)

    def language_model_loss(self, chars: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        forward_logits, backward_logits = self.language_model_logits(chars)
        forward_loss = F.cross_entropy(
            forward_logits[:, :-1].reshape(-1, forward_logits.size(-1)),
            token_ids[:, 1:].reshape(-1),
        )
        backward_loss = F.cross_entropy(
            backward_logits[:, 1:].reshape(-1, backward_logits.size(-1)),
            token_ids[:, :-1].reshape(-1),
        )
        return forward_loss + backward_loss


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(7)
    model = ELMo()
    chars = torch.randint(1, 262, (2, 8, 16))
    token_ids = torch.randint(0, 10_000, (2, 8))

    representations = model(chars)
    loss = model.language_model_loss(chars, token_ids)

    print("ELMo output:", tuple(representations.shape))
    print("LM loss:", round(loss.item(), 4))
    print("Parameters:", f"{count_parameters(model):,}")
