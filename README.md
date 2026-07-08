# PDF Q&A with Local RAG

A Retrieval-Augmented Generation system that lets you upload PDFs and ask questions.
Runs locally with Ollama or deploys to Streamlit Cloud with Google Gemini (free).

## How it works

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                           ↓
User Question → Semantic Search → Context Injection → LLM → Answer
```

## Local dev (Ollama)

**Prerequisites:** [Ollama](https://ollama.ai/) with models pulled:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

**Run:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (free)

1. Get a free Gemini API key at https://aistudio.google.com/apikey
2. Push this repo to GitHub
3. Go to https://streamlit.io/cloud → New app → select your repo
4. In Settings → Secrets, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Deploy — your app will be live at `yourapp.streamlit.app`

## Project structure

```
rag-demo/
├── app.py              # Streamlit UI
├── ingest.py           # PDF parsing + chunking + indexing
├── rag_engine.py       # Retrieval + generation
├── llm_client.py       # LLM abstraction (Gemini / Ollama)
├── embeddings.py       # Embedding provider (Gemini / Ollama)
├── vector_store.py     # ChromaDB wrapper
├── config.py           # Settings
├── .streamlit/
│   └── secrets.toml    # API key template
└── requirements.txt
```
