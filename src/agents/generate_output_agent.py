"""Generation agent: builds the prompt and calls the LLM for an answer."""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from ..helpers.logger import get_logger
from ..prompts.generate_output_prompt import generate_output_prompt

logger = get_logger(__name__)


def generate_output_agent(query: str, documents: list[Document]) -> str:
    """Generate an answer for a query grounded in the given documents.

    Args:
        query: The user question.
        documents: Retrieved documents used as context.

    Returns:
        The LLM-generated answer as a string.
    """
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = generate_output_prompt(query, documents)

    logger.info("Generating answer for query: '%s' with %d documents.", query, len(documents))

    return str(llm.invoke(prompt).content)
