import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        nn.init.uniform_(self.input_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, center_words, context_words, negative_words):
        center = self.input_embeddings(center_words)
        context = self.output_embeddings(context_words)
        negatives = self.output_embeddings(negative_words)

        positive_score = torch.sum(center * context, dim=1)
        negative_score = torch.bmm(negatives, center.unsqueeze(2)).squeeze(2)

        positive_loss = F.logsigmoid(positive_score)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=1)

        return -(positive_loss + negative_loss).mean()

    def embeddings(self):
        return self.input_embeddings.weight


if __name__ == "__main__":
    model = SkipGramNegativeSampling(vocab_size=10000, embedding_dim=300)
    center = torch.randint(0, 10000, (32,))
    context = torch.randint(0, 10000, (32,))
    negatives = torch.randint(0, 10000, (32, 5))
    print(model(center, context, negatives).item())
