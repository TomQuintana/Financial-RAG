"""OpenAI LLM client."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..settings import settings
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Builds a ChatOpenAI model from settings. Key is read from the env by langchain."""

    def get_llm(self) -> BaseChatModel:
        """Return a configured ChatOpenAI model."""
        return ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
