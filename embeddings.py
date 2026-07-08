import config


class GeminiEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=[text],
        )
        return result.embeddings[0].values


def get_embedding_function():
    if config.GEMINI_API_KEY:
        return GeminiEmbeddings(api_key=config.GEMINI_API_KEY)
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
