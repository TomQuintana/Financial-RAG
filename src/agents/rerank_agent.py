from langchain_core.documents import Document

from ..logger import get_logger

logger = get_logger(__name__)


def rerank_agent(documents: list[Document], scores: list[float]) -> tuple[list[Document], list[float]]:
    # ponytail: stub — add cross-encoder (e.g. ms-marco) when retrieval quality needs boost
    logger.debug("Reranking - not implemented yet")
    return documents, scores
