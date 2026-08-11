# Mini-RAG: Wikipedia Question-Answering Pipeline

A lightweight **Retrieval-Augmented Generation (RAG)** system built in Python. This project indexes Wikipedia passages and leverages vector search combined with a Large Language Model (LLM) to deliver precise, context-aware answers to user queries.

Developed as a step-by-step.

## Architecture Overview

The pipeline operates in two core stages:
1. **Indexing & Vectorization:** Ingestion of the Wikipedia corpus, generation of dense vector embeddings, and persistent storage in a local vector store.
2. **Retrieval & Generation:** Top-$k$ similarity retrieval based on user input, prompt augmentation with retrieved context, and response synthesis via an LLM.

```text
[Wikipedia Corpus] ──> [Embedding Model] ──> [Vector Store (ChromaDB/FAISS)]
                                                       │
[User Query] ───────> [Similarity Search] ────────────┘
                              │
                    [Retrieved Context]
                              │
                     [Prompt Assembly] ──> [LLM Engine] ──> [Final Answer]
