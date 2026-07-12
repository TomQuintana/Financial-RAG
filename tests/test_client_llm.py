"""Tests for the LLM client factory and provider selection."""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from src.client_llm.factory_llm import get_llm
from src.settings import settings


def test_get_llm_returns_chat_model(monkeypatch):
    """Verifies the factory builds a LangChain chat model for the default provider.

    A dummy key is injected so ChatOpenAI constructs offline (no network at init).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_llm.cache_clear()
    assert isinstance(get_llm(), BaseChatModel)
    get_llm.cache_clear()


def test_unknown_provider_raises(monkeypatch):
    """Verifies an unknown LLM_PROVIDER raises before building any client."""
    get_llm.cache_clear()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        get_llm()
    get_llm.cache_clear()
