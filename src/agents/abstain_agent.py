from ..helpers.logger import get_logger

logger = get_logger(__name__)

RELEVANCE_THRESHOLD = 0  # ponytail: cross-encoder logit, ajustar con eval


def abstain_agent(scores: list[float]) -> bool:
    if not scores:
        logger.debug("Abstain: no scores, abstaining")
        return True
    best = max(scores)
    abstain = best < RELEVANCE_THRESHOLD

    logger.debug(
        "Abstain: best score=%.2f, threshold=%s, abstain=%s",
        best,
        RELEVANCE_THRESHOLD,
        abstain,
    )

    return abstain
