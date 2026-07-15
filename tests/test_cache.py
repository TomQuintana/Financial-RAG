"""Tests for the semantic cache (src/cache.py)."""

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

import src.cache as cache

_VECTORS = {
    "what is apple revenue": [1.0, 0.0],
    "what was apple's revenue": [0.95, 0.05],
    "what is the weather today": [0.0, 1.0],
}


class _FakeEmbeddings(Embeddings):
    """Deterministic stand-in so cache tests don't hit the OpenAI API."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _VECTORS[text]


@pytest.fixture(autouse=True)
def fake_qa_cache(tmp_path, monkeypatch):
    """Swap the module-level qa_cache for an isolated, offline instance per test."""
    fake = Chroma(
        collection_name="qa_cache_test",
        embedding_function=_FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    monkeypatch.setattr(cache, "qa_cache", fake)


def test_get_cached_returns_none_when_empty():
    """An empty cache is always a miss, regardless of the query."""
    assert cache.get_cached("what is apple revenue") is None


def test_round_trip_exact_query():
    """A cached answer is returned for the exact same query."""
    cache.set_cached("what is apple revenue", "$416B")
    assert cache.get_cached("what is apple revenue") == "$416B"


def test_hit_on_close_paraphrase():
    """A semantically close paraphrase is still a hit."""
    cache.set_cached("what is apple revenue", "$416B")
    assert cache.get_cached("what was apple's revenue") == "$416B"


def test_miss_on_unrelated_query():
    """An unrelated query is a miss even when the cache has entries."""
    cache.set_cached("what is apple revenue", "$416B")
    assert cache.get_cached("what is the weather today") is None


def test_set_cached_skips_empty_answer():
    """Empty answers are never written to the cache."""
    cache.set_cached("what is apple revenue", "")
    assert cache.get_cached("what is apple revenue") is None


def test_set_cached_upserts_same_query():
    """Re-caching the same query overwrites the old answer instead of duplicating it."""
    cache.set_cached("what is apple revenue", "$416B")
    cache.set_cached("what is apple revenue", "$416,161M")

    assert cache.get_cached("what is apple revenue") == "$416,161M"
    assert cache.qa_cache._collection.count() == 1
