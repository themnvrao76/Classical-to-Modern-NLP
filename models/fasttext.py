import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTextSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, bucket_size=2000000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bucket_size = bucket_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.subword_embeddings = nn.Embedding(bucket_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.5 / self.embedding_dim
        nn.init.uniform_(self.word_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.subword_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

    def compose(self, word_ids, subword_ids, subword_mask=None):
        word = self.word_embeddings(word_ids)
        subwords = self.subword_embeddings(subword_ids)

        if subword_mask is None:
            subword_sum = subwords.sum(dim=-2)
            count = subwords.new_full(word.shape[:-1] + (1,), subwords.size(-2) + 1)
        else:
            mask = subword_mask.to(subwords.dtype).unsqueeze(-1)
            subword_sum = (subwords * mask).sum(dim=-2)
            count = mask.sum(dim=-2) + 1

        return (word + subword_sum) / count.clamp_min(1)

    def forward(self, center_ids, center_subwords, positive_ids, negative_ids, subword_mask=None):
        center = self.compose(center_ids, center_subwords, subword_mask)
        positive = self.output_embeddings(positive_ids)
        negative = self.output_embeddings(negative_ids)

        positive_score = (center * positive).sum(dim=-1)
        negative_score = torch.bmm(negative, center.unsqueeze(-1)).squeeze(-1)

        positive_loss = F.logsigmoid(positive_score)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=-1)
        return -(positive_loss + negative_loss).mean()


def character_ngrams(token, min_n=3, max_n=6):
    wrapped = f"<{token}>"
    return [
        wrapped[i : i + n]
        for n in range(min_n, max_n + 1)
        for i in range(len(wrapped) - n + 1)
    ]


def fnv1a_hash(text):
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def subword_buckets(token, bucket_size=2000000, min_n=3, max_n=6):
    return [fnv1a_hash(ngram) % bucket_size for ngram in character_ngrams(token, min_n, max_n)]


if __name__ == "__main__":
    torch.manual_seed(7)
    model = FastTextSkipGram(vocab_size=10000, embedding_dim=300, bucket_size=200000)

    tokens = ["vision", "language"]
    buckets = [subword_buckets(token, model.bucket_size) for token in tokens]
    width = max(map(len, buckets))
    subword_ids = torch.zeros(len(tokens), width, dtype=torch.long)
    mask = torch.zeros(len(tokens), width, dtype=torch.bool)
    for row, ids in enumerate(buckets):
        subword_ids[row, : len(ids)] = torch.tensor(ids)
        mask[row, : len(ids)] = True

    center_ids = torch.tensor([12, 91])
    positive_ids = torch.tensor([44, 17])
    negative_ids = torch.randint(0, model.vocab_size, (2, 5))
    loss = model(center_ids, subword_ids, positive_ids, negative_ids, mask)

    parameters = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"parameters: {parameters:,}")
    print(f"loss: {loss.item():.4f}")
