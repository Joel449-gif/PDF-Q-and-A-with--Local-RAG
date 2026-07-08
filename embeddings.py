import config
from langchain_community.embeddings import OllamaEmbeddings


def get_embedding_function():
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
