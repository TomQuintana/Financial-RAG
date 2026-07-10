"""Retrieval agent: similarity search against the ChromaDB vector store."""

from langchain_core.documents import Document

from ..helpers.logger import get_logger
from ..services.retrieve import load_vectorstore

logger = get_logger(__name__)


def retrieve_agent(query: str, k: int = 20) -> tuple[list[Document], list[float]]:
    """Retrieve the top-k documents for a query with their similarity scores.

    Args:
        query: The user question.
        k: Number of documents to retrieve.

    Returns:
        A tuple of (documents, scores); both lists are empty when there are
        no results.
    """
    vectors = load_vectorstore()
    results = vectors.similarity_search_with_score(query, k=k)

    logger.debug("Retrieved %d documents for query: '%s'", len(results), query)

    if not results:
        return [], []

    docs, scores = zip(*results, strict=True)

    return list(docs), list(scores)
