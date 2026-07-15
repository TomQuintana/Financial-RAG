"""Tests for GraphService.process_query response handling."""

from unittest.mock import MagicMock

import pytest

from src.graph.graph_service import GraphService


@pytest.fixture(autouse=True)
def fake_cache(monkeypatch):
    """Replace the cache with an in-memory dict local to each test.

    process_query calls get_cached/set_cached from src.cache; without this, these
    tests would hit the real persisted qa_cache (network calls + cross-test pollution
    since every test here uses the same query string).
    """
    store: dict[str, str] = {}
    monkeypatch.setattr("src.graph.graph_service.get_cached", store.get)
    monkeypatch.setattr(
        "src.graph.graph_service.set_cached",
        lambda query, answer: store.update({query: answer}) if answer else None,
    )


def make_service(invoke_return) -> GraphService:
    """Build a GraphService whose graph is mocked to return a fixed state.

    Args:
        invoke_return: The dict returned by ``graph.invoke``, standing in for
            the final RAGState the pipeline would produce.

    Returns:
        A GraphService instance ready to run ``process_query`` without hitting
        the real LLM or vector store.
    """
    service = GraphService()
    service.graph = MagicMock()
    service.graph.invoke.return_value = invoke_return
    return service


def test_returns_specific_message_when_abstaining():
    """When the state has ``abstain=True`` the abstain message is returned.

    The abstain path must be distinct from the empty-answer fallback, so it
    also asserts the generic fallback text is absent.
    """
    service = make_service({"answer": "", "abstain": True})
    result = service.process_query("¿Cuál es el clima?")

    assert result["success"] is True
    assert "No encontré información" in result["response"]
    assert "No se generó respuesta" not in result["response"]


def test_returns_answer_when_not_abstaining():
    """When ``abstain=False`` and an answer exists, the answer is returned."""
    service = make_service({"answer": "Apple revenue was $394B.", "abstain": False})
    result = service.process_query("Apple revenue?")

    assert result["success"] is True
    assert result["response"] == "Apple revenue was $394B."


def test_returns_fallback_when_answer_empty_without_abstain():
    """When ``abstain=False`` but the answer is empty, the fallback is used."""
    service = make_service({"answer": "", "abstain": False})
    result = service.process_query("Apple revenue?")

    assert result["success"] is True
    assert result["response"] == "No se generó respuesta."


def test_returns_error_when_graph_raises():
    """When ``graph.invoke`` raises, the error is caught and reported."""
    service = GraphService()
    service.graph = MagicMock()
    service.graph.invoke.side_effect = RuntimeError("boom")
    result = service.process_query("Apple revenue?")

    assert result["success"] is False
    assert result["error"] == "boom"
    assert "Error inesperado" in result["response"]
