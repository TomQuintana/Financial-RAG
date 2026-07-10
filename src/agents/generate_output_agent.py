from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from ..helpers.logger import get_logger
from ..prompts.generate_output_prompt import generate_output_prompt

logger = get_logger(__name__)


def generate_output_agent(query: str, documents: list[Document]) -> str:
    # context = "\n\n".join(document.page_content for document in documents)

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = generate_output_prompt(query, documents)

    logger.info("Generating answer for query: '%s' with %d documents.", query, len(documents))

    return str(llm.invoke(prompt).content)
