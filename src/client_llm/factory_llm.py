"""LLM client package: provider registry and a cached factory."""

from functools import cache

from langchain_core.language_models.chat_models import BaseChatModel

from ..helpers.logger import get_logger
from ..settings import settings
from .base import BaseLLMClient
from .openai_client import OpenAIClient

_PROVIDERS: dict[str, type[BaseLLMClient]] = {"openai": OpenAIClient}  # add "anthropic" later

logger = get_logger(__name__)


# ponytail: one client per process so FastAPI reuses the HTTP pool.
# Drop @cache if per-request model/temperature is ever needed (settings read fresh each call).
@cache
def get_llm() -> BaseChatModel:
    """Build (once) and return the chat model for the configured LLM_PROVIDER.

    Returns:
        A configured LangChain chat model.

    Raises:
        ValueError: If LLM_PROVIDER is not a known provider.
    """
    provider = settings.LLM_PROVIDER
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    logger.debug("Building LLM client for provider %s", provider)
    return _PROVIDERS[provider]().get_llm()
