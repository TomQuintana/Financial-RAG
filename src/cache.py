"""Query/response caching layer between the API and the RAG pipeline."""

import hashlib
import time
from functools import cache

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .helpers.logger import get_logger
from .services.ingest import CHROMA_DIR, EMBEDDING_MODEL

logger = get_logger(__name__)

# TODO: move to constant folder
CACHE_DISTANCE_THRESHOLD = 0.14
CACHE_TTL_SECONDS = 60 * 60 * 24


@cache
def load_cache() -> Chroma:
    """Connect (lazily) to the qa_cache ChromaDB collection.

    Memoized with ``@cache`` so the store and embeddings client are built once,
    on first use, instead of at import time.

    Returns:
        The shared ``Chroma`` instance backing the query/response cache.
    """
    return Chroma(
        collection_name="qa_cache",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_cached(query: str) -> str | None:
    """Return a cached answer for a semantically similar prior query, if any.

    Args:
        query: The incoming user question.

    Returns:
        The cached answer, or None on a miss or an empty cache.
    """
    normalized = query.strip().lower()

    if load_cache()._collection.count() == 0:
        return None

    doc, distance = load_cache().similarity_search_with_score(normalized, k=1)[0]

    if distance > CACHE_DISTANCE_THRESHOLD:
        return None

    age = time.time() - doc.metadata.get("cached_at", 0)
    if age > CACHE_TTL_SECONDS:
        logger.info("Cache hit but expired (age=%.0fs) for query: %r", age, query)
        return None

    logger.info("Cache hit (distance=%.4f) for query: %r", distance, query)
    return doc.metadata.get("answer")


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

    load_cache().add_texts(
        texts=[normalized],
        metadatas=[{"answer": answer, "cached_at": time.time()}],
        ids=[cache_id],
    )


def invalidate_cache() -> None:
    """Drop all cached answers. Call after reindexing the corpus."""
    store = load_cache()
    ids = store.get()["ids"]
    if ids:
        store.delete(ids=ids)
        logger.info("Cache invalidated (%d entries dropped)", len(ids))
