"""Contract for LLM provider clients."""

from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel


class BaseLLMClient(ABC):
    """Base contract every LLM provider client must satisfy."""

    # ponytail: ABC holds the OpenAI + planned Anthropic providers to one interface.

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Return a configured LangChain chat model."""
