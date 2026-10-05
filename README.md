<h1 align="center">Classical to Modern NLP</h1>

<p align="center">
  <strong>A PyTorch reference for the architectures and ideas that shaped Natural Language Processing.</strong>
</p>

<p align="center">
  From word embeddings and recurrent networks to Transformers, pretrained language models, retrieval, PEFT, and modern LLM systems.
</p>

<p align="center">
  <strong>PyTorch</strong> · <strong>26 implementation files</strong> · <strong>Original papers</strong> · <strong>Active development</strong>
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

> **Recurrent sequence models** → **Gated memory** → **Distributed representations** → **Subword representations** → **Seq2Seq** → **Attention** → **Transformer** → **Contextual pretraining** → **Large language models** → **Retrieval & parameter-efficient adaptation**

| Era | Key Development | Why It Mattered |
|---|---|---|
| 1990–1997 | Elman RNN, LSTM | Recurrent state followed by gated memory for learning longer dependencies |
| 2013–2016 | Word2Vec, GloVe, GRU, FastText | Dense vectors, compact gated recurrence, and morphology-aware representations |
| 2014 | Seq2Seq | End-to-end neural sequence transduction with separate encoder and decoder recurrent networks |
| 2014–2015 | Bahdanau Attention | Learned soft alignment let the decoder dynamically retrieve relevant encoder states instead of relying on one fixed context vector |
| 2017 | Transformer | Replaced recurrence with parallel multi-head self-attention, positional encoding, and feed-forward blocks |
| 2018 | ELMo | Deep bidirectional language models produced context-sensitive token representations from character-aware inputs |
| 2018–2019 | GPT, BERT | Large-scale Transformer language-model pretraining |
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
| **Transformer — Encoder–Decoder Self-Attention** | 2017 | 59,508,496 (10k source/target vocab, 512d, 8 heads, 6+6 layers demo) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [`models/transformer.py`](models/transformer.py) |
| **Mixture-of-Experts — Sparse Routed FFN** | 2017 | Config-dependent | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | [`models/mixture_of_experts.py`](models/mixture_of_experts.py) |
| **ELMo — Deep Contextualized Word Representations** | 2018 | 8,016,996 (10k vocab, 512d token representation, 2-layer BiLSTM demo) | [Deep contextualized word representations](https://aclanthology.org/N18-1202/) | [`models/elmo.py`](models/elmo.py) |
| **GPT — Generative Pre-Training** | 2018 | 100,807,680 (40k vocab, 512-token context, 768d, 12 heads, 12 layers demo) | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | [`models/gpt.py`](models/gpt.py) |
| **BERT — Bidirectional Encoder Representations from Transformers** | 2018 | 109,514,298 (BERT-Base: 30,522 vocab, 768d, 12 heads, 12 layers) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | [`models/bert.py`](models/bert.py) |
| **GPT-2 — Language Models are Unsupervised Multitask Learners** | 2019 | 124,439,808 (GPT-2 small: 50,257 vocab, 1024 context, 768d, 12 heads, 12 layers) | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [`models/gpt2.py`](models/gpt2.py) |
| **Transformer-XL — Segment Recurrence + Relative Attention** | 2019 | Config-dependent | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) | [`models/transformer_xl.py`](models/transformer_xl.py) |
| **Sentence-BERT — Sentence Embeddings** | 2019 | Config-dependent | [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) | [`models/sentence_bert.py`](models/sentence_bert.py) |
| **XLNet — Permutation Language Modeling** | 2019 | Config-dependent | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) | [`models/xlnet.py`](models/xlnet.py) |
| **RoBERTa — Robustly Optimized BERT Pretraining** | 2019 | Config-dependent | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | [`models/roberta.py`](models/roberta.py) |
| **ALBERT — Parameter-Sharing Transformer Encoder** | 2019 | Config-dependent | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) | [`models/albert.py`](models/albert.py) |
| **DistilBERT — Distilled Transformer Encoder** | 2019 | Config-dependent | [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) | [`models/distilbert.py`](models/distilbert.py) |
| **Modern LLM Components — RMSNorm, RoPE, GQA, KV Cache** | 2019–2023 | Config-dependent | [RMSNorm](https://arxiv.org/abs/1910.07467) · [RoPE](https://arxiv.org/abs/2104.09864) · [GQA](https://arxiv.org/abs/2305.13245) | [`models/modern_llm_components.py`](models/modern_llm_components.py) |
| **T5 — Text-to-Text Transfer Transformer** | 2020 | Config-dependent | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | [`models/t5.py`](models/t5.py) |
| **BART — Denoising Seq2Seq Pretraining** | 2020 | Config-dependent | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation](https://arxiv.org/abs/1910.13461) | [`models/bart.py`](models/bart.py) |
| **RAG — Retrieval-Augmented Generation** | 2020 | Config-dependent | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | [`models/rag.py`](models/rag.py) |
| **ELECTRA — Replaced Token Detection** | 2020 | Config-dependent | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) | [`models/electra.py`](models/electra.py) |
| **LoRA — Low-Rank Adaptation** | 2021 | Config-dependent | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | [`models/lora.py`](models/lora.py) |

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
python models/gpt2.py
```

---

## Roadmap

The repository will grow chronologically while covering the major branches of modern NLP.

**Foundations**  
Elman RNN → LSTM → Word2Vec → GloVe → GRU → FastText

**Sequence-to-Sequence & Attention**  
Seq2Seq ✓ → Bahdanau Attention ✓ → Transformer ✓

**Pretrained Language Models**  
ELMo ✓ → GPT ✓ → BERT ✓ → GPT-2 ✓ → RoBERTa ✓ → ALBERT ✓ → XLNet ✓ → Transformer-XL ✓

**Text-to-Text & Efficient Transformers**  
T5 ✓ → BART ✓ → ELECTRA ✓ → DistilBERT ✓

**Representation & Adaptation**  
Sentence-BERT ✓ → contrastive sentence learning → adapters → LoRA ✓

**Retrieval & Modern LLM Systems**  
Dense retrieval ✓ → RAG ✓ → RoPE ✓ → RMSNorm ✓ → grouped-query attention ✓ → KV caching ✓ → Mixture-of-Experts ✓

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
