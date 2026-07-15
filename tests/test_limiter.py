"""Tests for the /query rate limit (src/limiter.py + api/main.py wiring)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.limiter import limiter


@pytest.fixture(autouse=True)
def reset_limiter():
    """Slowapi's default storage is in-memory and process-wide — reset between tests."""
    limiter.reset()


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    """Stub out the real RAG pipeline; this file only tests the rate limit."""
    monkeypatch.setattr(
        "api.main.graph_service.process_query",
        lambda query: {"response": "ok", "success": True, "error": None, "metadata": {}},
    )


def test_requests_within_limit_return_200():
    """Requests up to the configured limit succeed normally."""
    client = TestClient(app)
    payload = {"query": "irrelevant"}

    for _ in range(10):  # matches RATE_LIMIT = "10/minute" default
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        assert response.json()["response"] == "ok"


def test_request_over_limit_returns_429():
    """The request past the limit is rejected with a clear message."""
    client = TestClient(app)
    payload = {"query": "irrelevant"}

    for _ in range(10):
        client.post("/query", json=payload)

    response = client.post("/query", json=payload)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]
