"""Central settings sourced from environment variables (via load_dotenv)."""

import os

from dotenv import load_dotenv

load_dotenv()  # populate os.environ once; never read .env directly


class Settings:
    """App configuration read from the environment.

    Attributes:
        OPENAI_API_KEY: OpenAI key (read by langchain implicitly; never log it).
        LLM_PROVIDER: Which client_llm provider to build (e.g. ``openai``).
        LLM_MODEL: Chat model name.
        LLM_TEMPERATURE: Sampling temperature; 0 for anti-hallucination.
        LOG_LEVEL: Comma-separated logging levels.
    """

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # ponytail: pydantic-settings is the upgrade path if validation/nesting is ever needed.


settings = Settings()
