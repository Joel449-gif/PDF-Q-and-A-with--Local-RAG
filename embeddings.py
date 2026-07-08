import config


def get_embedding_function():
    if config.GEMINI_API_KEY:
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
