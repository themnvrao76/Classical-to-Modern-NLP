import torch
import torch.nn as nn


class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim, x_max=100.0, alpha=0.75):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)
        self.x_max = x_max
        self.alpha = alpha

        scale = 0.5 / embedding_dim
        nn.init.uniform_(self.word_embeddings.weight, -scale, scale)
        nn.init.uniform_(self.context_embeddings.weight, -scale, scale)
        nn.init.zeros_(self.word_biases.weight)
        nn.init.zeros_(self.context_biases.weight)

    def weighting(self, cooccurrences):
        return torch.where(
            cooccurrences < self.x_max,
            (cooccurrences / self.x_max).pow(self.alpha),
            torch.ones_like(cooccurrences),
        )

    def forward(self, word_ids, context_ids, cooccurrences):
        if torch.any(cooccurrences <= 0):
            raise ValueError("GloVe requires strictly positive co-occurrence counts")

        word = self.word_embeddings(word_ids)
        context = self.context_embeddings(context_ids)
        word_bias = self.word_biases(word_ids).squeeze(-1)
        context_bias = self.context_biases(context_ids).squeeze(-1)

        prediction = (word * context).sum(dim=-1) + word_bias + context_bias
        target = torch.log(cooccurrences)
        weights = self.weighting(cooccurrences)
        return (weights * (prediction - target).pow(2)).mean()

    def embeddings(self):
        return self.word_embeddings.weight + self.context_embeddings.weight


if __name__ == "__main__":
    model = GloVe(vocab_size=10000, embedding_dim=300)
    words = torch.randint(0, 10000, (64,))
    contexts = torch.randint(0, 10000, (64,))
    counts = torch.randint(1, 200, (64,)).float()
    loss = model(words, contexts, counts)
    print(f"loss={loss.item():.4f}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
