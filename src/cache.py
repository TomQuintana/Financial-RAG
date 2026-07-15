"""Query/response caching layer between the API and the RAG pipeline."""

import hashlib
import time

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .helpers.logger import get_logger
from .services.ingest import CHROMA_DIR, EMBEDDING_MODEL

logger = get_logger(__name__)

CACHE_DISTANCE_THRESHOLD = 0.15


qa_cache = Chroma(
    collection_name="qa_cache",
    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=str(CHROMA_DIR),
)


def get_cached(query: str) -> str | None:
    """Return a cached answer for a semantically similar prior query, if any.

    Args:
        query: The incoming user question.

    Returns:
        The cached answer, or None on a miss or an empty cache.
    """
    normalized = query.strip().lower()

    if qa_cache._collection.count() == 0:
        return None

    doc, distance = qa_cache.similarity_search_with_score(normalized, k=1)[0]

    if distance <= CACHE_DISTANCE_THRESHOLD:
        logger.info("Cache hit (distance=%.4f) for query: %r", distance, query)
        return doc.metadata.get("answer")

    return None


def set_cached(query: str, answer: str) -> None:
    """Store an answer in the cache, keyed by the normalized query.

    Args:
        query: The user question that produced ``answer``.
        answer: The generated answer. Empty/falsy answers are not cached.
    """
    if not answer:
        return

    normalized = query.strip().lower()
    cache_id = hashlib.md5(normalized.encode()).hexdigest()

    qa_cache.add_texts(
        texts=[normalized],
        metadatas=[{"answer": answer, "cached_at": time.time()}],
        ids=[cache_id],
    )
