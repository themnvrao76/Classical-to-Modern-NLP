<h1 align="center">Classical to Modern NLP</h1>

<p align="center">
  <strong>A PyTorch reference for the architectures and ideas that shaped Natural Language Processing.</strong>
</p>

<p align="center">
  From word embeddings and recurrent networks to Transformers, pretrained language models, retrieval, PEFT, and modern LLM systems.
</p>

<p align="center">
  <strong>PyTorch</strong> · <strong>8 implemented methods</strong> · <strong>Original papers</strong> · <strong>Active development</strong>
</p>

---

## Overview

**Classical to Modern NLP** is a growing collection of implementations of influential natural language processing architectures and methods. The repository follows the technical evolution of NLP chronologically, connecting each implementation to the research paper or idea that introduced it.

The aim is to provide a compact reference for understanding **how modern language models evolved** rather than a collection of unrelated model files.

### What this repository covers

| Area | Representative Methods |
|---|---|
| Word Representations | Word2Vec, GloVe, FastText |
| Sequence Modeling | RNN, LSTM, GRU |
| Neural Machine Translation | Seq2Seq, Bahdanau Attention |
| Transformer Foundations | Transformer, Transformer-XL |
| Pretrained Language Models | ELMo, BERT, GPT, GPT-2, RoBERTa |
| Encoder-Decoder Models | T5, BART |
| Efficient NLP | ALBERT, ELECTRA, DistilBERT |
| Sentence Representation | Sentence-BERT, contrastive learning |
| Parameter-Efficient Training | Adapters, LoRA / PEFT |
| Retrieval | Dense retrieval, RAG |
| Modern LLM Systems | decoder-only Transformers, RoPE, RMSNorm, KV cache, MoE |

---

## Evolution of NLP

> **Recurrent sequence models** → **Gated memory** → **Distributed representations** → **Subword representations** → **Seq2Seq** → **Attention** → **Transformer** → **Pretraining** → **Large language models** → **Retrieval & parameter-efficient adaptation**

| Era | Key Development | Why It Mattered |
|---|---|---|
| 1990–1997 | Elman RNN, LSTM | Recurrent state followed by gated memory for learning longer dependencies |
| 2013–2016 | Word2Vec, GloVe, GRU, FastText | Dense vectors, compact gated recurrence, and morphology-aware representations |
| 2014 | Seq2Seq | End-to-end neural sequence transduction with separate encoder and decoder recurrent networks |
| 2014–2015 | Bahdanau Attention | Learned soft alignment let the decoder dynamically retrieve relevant encoder states instead of relying on one fixed context vector |
| 2017 | Transformer | Replaced recurrence with self-attention |
| 2018–2019 | ELMo, GPT, BERT | Large-scale language-model pretraining |
| 2019–2020 | GPT-2, RoBERTa, T5, BART | Scaling and unified text-to-text learning |
| 2020–2022 | Dense retrieval, scaling, instruction methods | Stronger retrieval and general-purpose language models |
| 2021–present | LoRA, RAG, MoE, modern LLM components | Efficient adaptation and scalable LLM systems |

---

## Implemented

| Model / Method | Year | Parameters | Original Paper | Implementation |
|---|---:|---:|---|---|
| **Elman RNN — Simple Recurrent Network** | 1990 | 8,083,728 (10k vocab, 256d embedding, 512 hidden demo) | [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1) | [`models/rnn.py`](models/rnn.py) |
| **LSTM — Long Short-Term Memory** | 1997 | 9,264,912 (10k vocab, 256d embedding, 512 hidden demo) | [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) | [`models/lstm.py`](models/lstm.py) |
| **Word2Vec — Skip-gram + Negative Sampling** | 2013 | Vocabulary-dependent | [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) | [`models/word2vec_skipgram.py`](models/word2vec_skipgram.py) |
| **GloVe — Global Vectors for Word Representation** | 2014 | 6,020,000 (10k vocab, 300d demo) | [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) | [`models/glove.py`](models/glove.py) |
| **GRU — Gated Recurrent Unit** | 2014 | 8,871,184 (10k vocab, 256d embedding, 512 hidden demo) | [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078) | [`models/gru.py`](models/gru.py) |
| **Seq2Seq — LSTM Encoder–Decoder** | 2014 | 17,598,224 (10k source/target vocab, 256d embedding, 512 hidden, 2-layer demo) | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | [`models/seq2seq.py`](models/seq2seq.py) |
| **Bahdanau Attention — Additive Neural Attention** | 2014 | 24,292,112 (10k source/target vocab, 256d embedding, 512 hidden/attention, 2-layer demo) | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | [`models/bahdanau_attention.py`](models/bahdanau_attention.py) |
| **FastText — Subword Skip-gram** | 2016 | 66,000,000 (10k vocab, 300d, 200k-bucket demo) | [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/) | [`models/fasttext.py`](models/fasttext.py) |

> Parameter counts refer to the implementations in this repository. Methods whose size depends on vocabulary, embedding dimension, bucket size, or runtime configuration are marked accordingly.

---

## Quick Start

```bash
git clone https://github.com/themnvrao76/Classical-to-Modern-NLP.git
cd Classical-to-Modern-NLP
pip install torch
```

Run an implementation directly:

```bash
python models/bahdanau_attention.py
```

---

## Roadmap

The repository will grow chronologically while covering the major branches of modern NLP.

**Foundations**  
Elman RNN → LSTM → Word2Vec → GloVe → GRU → FastText

**Sequence-to-Sequence & Attention**  
Seq2Seq ✓ → Bahdanau Attention ✓ → Transformer

**Pretrained Language Models**  
ELMo → GPT → BERT → GPT-2 → RoBERTa → ALBERT → XLNet → Transformer-XL

**Text-to-Text & Efficient Transformers**  
T5 → BART → ELECTRA → DistilBERT

**Representation & Adaptation**  
Sentence-BERT → contrastive sentence learning → adapters → LoRA

**Retrieval & Modern LLM Systems**  
Dense retrieval → RAG → RoPE → RMSNorm → grouped-query attention → KV caching → Mixture-of-Experts

---

## Design Principles

Implementations in this repository aim to be:

- **Paper-oriented** — tied to influential research rather than arbitrary model selection
- **Readable** — focused on the architectural idea without unnecessary framework abstraction
- **Runnable** — implementations include lightweight checks where practical
- **Comparable** — year, parameter information, paper, and source code are indexed consistently
- **Progressive** — additions follow the historical development of NLP

---

## Topics

`natural-language-processing` · `deep-learning` · `pytorch` · `transformers` · `attention` · `bert` · `gpt` · `large-language-models` · `word-embeddings` · `rag` · `lora` · `peft` · `research-papers`

---

## References

Every implemented architecture or method is linked to its original paper or primary technical source in the table above.

<p align="center">
  <strong>From recurrent sequence models and distributed representations to modern large language model systems.</strong>
</p>
