from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from ..logger import get_logger

logger = get_logger(__name__)


def generate_output_agent(query: str, documents: list[Document]) -> str:
    context = "\n\n".join(d.page_content for d in documents)
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    logger.info("Generating answer for query: '%s' with %d documents.", query, len(documents))

    return str(llm.invoke(prompt).content)
