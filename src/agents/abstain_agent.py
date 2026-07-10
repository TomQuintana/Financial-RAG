"""Abstention agent: decides whether the pipeline answers or abstains."""

from ..helpers.logger import get_logger

RELEVANCE_THRESHOLD = 0.0

logger = get_logger(__name__)


def abstain_agent(scores: list[float]) -> bool:
    """Decide whether to abstain based on document relevance scores.

    Args:
        scores: Relevance logits from the reranker (CrossEncoder).

    Returns:
        True if there are no documents or the best score does not exceed
        the relevance threshold; False otherwise.
    """
    # ponytail: stub — define threshold when eval data is available

    if not scores:
        return True

    is_abstain = max(scores) <= RELEVANCE_THRESHOLD

    return is_abstain
