from langchain_core.documents import Document

from ..helpers.logger import get_logger
from ..retrieve import load_vectorstore

logger = get_logger(__name__)


def retrieve_agent(query: str, k: int = 20) -> tuple[list[Document], list[float]]:
    vectors = load_vectorstore()
    results = vectors.similarity_search_with_score(query, k=k)

    logger.debug("Retrieved %d documents for query: '%s'", len(results), query)

    if not results:
        return [], []

    docs, scores = zip(*results)

    return list(docs), list(scores)
