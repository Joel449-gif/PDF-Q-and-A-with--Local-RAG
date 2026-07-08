import config

_use_gemini = bool(config.GEMINI_API_KEY)


def get_embedding_function():
    if _use_gemini:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=config.GEMINI_API_KEY,
        )
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
