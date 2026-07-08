# PDF Q&A with Local RAG

A Retrieval-Augmented Generation system that lets you upload PDFs, TXTs, and Markdown files and ask questions. Runs entirely locally with Ollama.

## How it works

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                            ↓
User Question → Semantic Search → Context Injection → LLM → Answer
```

## Prerequisites

[Ollama](https://ollama.ai/) with these models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- PDF / TXT / Markdown support
- Streaming responses
- Source citation with relevance scores
- Per-file document management
- LRU question cache (invalidates on re-index)
- Prompt injection hardening (7-rule system prompt)
- Modern dark-theme UI

## Project structure

```
rag-demo/
├── app.py           # Streamlit UI
├── ingest.py        # PDF parsing + chunking + indexing
├── rag_engine.py    # Retrieval + generation
├── llm_client.py    # Ollama LLM client
├── embeddings.py    # Embedding provider
├── vector_store.py  # ChromaDB wrapper
├── config.py        # Settings
└── requirements.txt
```
