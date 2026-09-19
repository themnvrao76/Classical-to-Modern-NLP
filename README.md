# Classical to Modern NLP

<p align="center">
  <strong>Implementations of influential natural language processing architectures and methods — from word embeddings and recurrent networks to Transformers, large language models, retrieval, and modern NLP systems.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-NLP-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" alt="Active Development">
  <img src="https://img.shields.io/badge/Papers-Original%20Sources-success" alt="Original Papers">
</p>

## About

This repository follows the evolution of natural language processing from classical distributed representations and sequence models to attention, Transformers, language models, parameter-efficient fine-tuning, retrieval-augmented generation, mixture-of-experts, and modern LLM systems.

The focus is on readable implementations of historically influential and practically important methods, with links to the original papers and concise metadata for each addition.

## NLP Evolution

**Word Embeddings → RNN/LSTM/GRU → Seq2Seq & Attention → Transformer → ELMo/BERT/GPT → T5/BART → Efficient Transformers → PEFT/LoRA → Retrieval/RAG → MoE → Modern LLM Components**

## Model & Method Index

| Model / Method | Year | Parameters | Original Paper | Implementation |
|---|---:|---:|---|---|
| Word2Vec Skip-gram + Negative Sampling | 2013 | Depends on vocabulary / embedding size | [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) | [`models/word2vec_skipgram.py`](models/word2vec_skipgram.py) |

## Planned Coverage

- Word2Vec, GloVe, FastText
- RNN, LSTM, GRU
- Seq2Seq and attention
- Transformer
- ELMo, BERT, GPT, GPT-2, RoBERTa, ALBERT
- XLNet, Transformer-XL, T5, BART, ELECTRA, DistilBERT
- Tokenization and subword methods
- LoRA and parameter-efficient fine-tuning
- Retrieval-augmented generation
- Mixture-of-Experts
- Sentence embeddings and contrastive NLP
- Modern decoder-only LLM components

## Repository Topics

`natural-language-processing` · `nlp` · `pytorch` · `transformers` · `bert` · `gpt` · `large-language-models` · `rag` · `lora` · `attention` · `research-papers`
