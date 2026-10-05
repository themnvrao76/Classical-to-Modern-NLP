import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseRetriever(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.projection = nn.Linear(d_model, d_model)

    def encode(self, ids):
        mask = ids.ne(0).unsqueeze(-1)
        x = self.embedding(ids)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(self.projection(pooled), dim=-1)

    def forward(self, query_ids, passage_ids):
        q = self.encode(query_ids)
        p = self.encode(passage_ids)
        return q @ p.t()


class RAGGenerator(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256, layers=4, heads=8, max_position=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_position, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, heads, d_model * 4,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, decoder_ids, retrieved_memory):
        pos = torch.arange(decoder_ids.shape[1], device=decoder_ids.device)
        x = self.token(decoder_ids) + self.position(pos)[None]
        seq = x.shape[1]
        causal = torch.full((seq, seq), float("-inf"), device=x.device)
        causal = torch.triu(causal, diagonal=1)
        x = self.decoder(x, retrieved_memory, tgt_mask=causal)
        return self.head(self.norm(x))


class RAG(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256, layers=4, heads=8):
        super().__init__()
        self.retriever = DenseRetriever(vocab_size, d_model)
        self.generator = RAGGenerator(vocab_size, d_model, layers, heads)
        self.memory_projection = nn.Linear(d_model, d_model)

    def retrieve(self, query_ids, passage_ids, top_k=2):
        q = self.retriever.encode(query_ids)
        p = self.retriever.encode(passage_ids)
        scores = q @ p.t()
        values, indices = scores.topk(min(top_k, passage_ids.shape[0]), dim=-1)
        passage_vectors = p[indices]
        return values, self.memory_projection(passage_vectors)

    def forward(self, query_ids, passage_ids, decoder_ids, top_k=2):
        scores, memory = self.retrieve(query_ids, passage_ids, top_k)
        return self.generator(decoder_ids, memory), scores


if __name__ == "__main__":
    model = RAG(vocab_size=10000, d_model=256)
    query = torch.randint(1, 10000, (2, 16))
    passages = torch.randint(1, 10000, (8, 32))
    decoder = torch.randint(1, 10000, (2, 12))
    logits, scores = model(query, passages, decoder)
    print("logits:", tuple(logits.shape), "retrieval:", tuple(scores.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
