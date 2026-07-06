from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ..helpers.logger import get_logger

logger = get_logger(__name__)

_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = 4


def rerank_agent(
    query: str, documents: list[Document]
) -> tuple[list[Document], list[float]]:
    """Rerank documents using a cross-encoder running locally via HuggingFace.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, cached in ~/.cache/huggingface).
    No API key required — inference runs on local CPU.
    """
    if not documents:
        return [], []

    pairs = [[query, doc.page_content] for doc in documents]
    scores = _model.predict(pairs).tolist()
    ranked = sorted(
        zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True
    )
    docs, rerank_scores = zip(*ranked[:TOP_K], strict=True)

    logger.debug(
        "Reranked %d → %d docs, top score: %.2f",
        len(documents),
        TOP_K,
        rerank_scores[0],
    )

    return list(docs), list(rerank_scores)
