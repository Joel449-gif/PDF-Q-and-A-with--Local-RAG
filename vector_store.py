import chromadb
from langchain_chroma import Chroma
import config
from embeddings import get_embedding_function


_doc_version = 0


def get_vector_store():
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=config.CHROMA_DB_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_documents(documents):
    global _doc_version
    store = get_vector_store()
    store.add_documents(documents)
    _doc_version += 1


def similarity_search(query, k=config.TOP_K):
    store = get_vector_store()
    return store.similarity_search_with_relevance_scores(query, k=k)


def get_collection_stats():
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def list_documents():
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
        data = collection.get(include=["metadatas"])
        sources = {}
        for meta in (data.get("metadatas") or []):
            if meta and "source" in meta:
                name = meta["source"]
                sources[name] = sources.get(name, 0) + 1
        return [{"name": k, "chunks": v} for k, v in sorted(sources.items())]
    except Exception:
        return []


def delete_document(source_name: str):
    global _doc_version
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
        collection.delete(where={"source": source_name})
        _doc_version += 1
    except Exception:
        pass


def delete_all():
    global _doc_version
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
        _doc_version += 1
    except Exception:
        pass


def get_doc_version():
    return _doc_version
