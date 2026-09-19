<h1 align="center">Classical to Modern NLP</h1>

<p align="center">
  <strong>A PyTorch reference for the architectures and ideas that shaped Natural Language Processing.</strong>
</p>

<p align="center">
  From word embeddings and recurrent networks to Transformers, pretrained language models, retrieval, PEFT, and modern LLM systems.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-NLP-orange?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Methods-1-blue" alt="Methods">
  <img src="https://img.shields.io/badge/Papers-Original%20Sources-green" alt="Papers">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
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

> **Distributed representations** → **Recurrent networks** → **Seq2Seq** → **Attention** → **Transformer** → **Pretraining** → **Large language models** → **Retrieval & parameter-efficient adaptation**

| Era | Key Development | Why It Mattered |
|---|---|---|
| 2013–2014 | Word2Vec, GloVe | Dense distributed word representations |
| 2014–2016 | LSTM, GRU, Seq2Seq | Neural sequence modeling and translation |
| 2015–2017 | Neural Attention | Dynamic focus over input sequences |
| 2017 | Transformer | Replaced recurrence with self-attention |
| 2018–2019 | ELMo, GPT, BERT | Large-scale language-model pretraining |
| 2019–2020 | GPT-2, RoBERTa, T5, BART | Scaling and unified text-to-text learning |
| 2020–2022 | Dense retrieval, scaling, instruction methods | Stronger retrieval and general-purpose language models |
| 2021–present | LoRA, RAG, MoE, modern LLM components | Efficient adaptation and scalable LLM systems |

---

## Implemented

| Model / Method | Year | Parameters | Original Paper | Implementation |
|---|---:|---:|---|---|
| **Word2Vec — Skip-gram + Negative Sampling** | 2013 | Vocabulary-dependent | [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) | [`models/word2vec_skipgram.py`](models/word2vec_skipgram.py) |

> Parameter counts refer to the implementations in this repository. Methods whose size depends on vocabulary, embedding dimension, or runtime configuration are marked accordingly.

---

## Quick Start

```bash
git clone https://github.com/themnvrao76/Classical-to-Modern-NLP.git
cd Classical-to-Modern-NLP
pip install torch
```

Run an implementation directly:

```bash
python models/word2vec_skipgram.py
```

---

## Roadmap

The repository will grow chronologically while covering the major branches of modern NLP.

**Foundations**  
Word2Vec → GloVe → FastText → RNN → LSTM → GRU

**Sequence-to-Sequence & Attention**  
Seq2Seq → Bahdanau Attention → Transformer

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
  <strong>From distributed word representations to modern large language model systems.</strong>
</p>
