import logging
import time

import config
import llm_client
from vector_store import similarity_search, get_doc_version

logger = logging.getLogger(__name__)


def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc, _ in context_chunks
    )
    return (
        f"{config.SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def stream_ask(query: str):
    """Generator yielding (text_token, sources_or_None)."""
    results = similarity_search(query)
    if not results:
        yield ("No documents indexed. Upload a file first.", None)
        return

    prompt = build_prompt(query, results)
    start = time.time()

    for token in llm_client.generate_stream(prompt):
        yield (token, None)

    elapsed = time.time() - start
    logger.info("Generated in %.2fs for query: %.60s", elapsed, query)

    sources = [
        {
            "content": doc.page_content[:500],
            "source": doc.metadata.get("source", "unknown"),
            "score": round(score, 3),
        }
        for doc, score in results
    ]
    yield (None, sources)


def ask(query: str):
    results = similarity_search(query)
    if not results:
        return None, []

    prompt = build_prompt(query, results)
    answer = llm_client.generate(prompt)

    sources = [
        {
            "content": doc.page_content[:500],
            "source": doc.metadata.get("source", "unknown"),
            "score": round(score, 3),
        }
        for doc, score in results
    ]
    return answer, sources


_question_cache: dict[str, tuple[str, list]] = {}
_last_cache_version = -1


def ask_cached(query: str):
    global _last_cache_version
    current_version = get_doc_version()

    if current_version != _last_cache_version:
        _question_cache.clear()
        _last_cache_version = current_version

    key = query.lower().strip()
    if key in _question_cache:
        logger.info("Cache hit for: %.60s", query)
        return _question_cache[key]

    logger.info("Cache miss for: %.60s", query)
    answer, sources = ask(query)
    if answer and sources:
        if len(_question_cache) < config.CACHE_MAXSIZE:
            _question_cache[key] = (answer, sources)
    return answer, sources
