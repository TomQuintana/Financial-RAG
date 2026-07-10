from ..helpers.logger import get_logger

RELEVANCE_THRESHOLD = 0.0

logger = get_logger(__name__)


def abstain_agent(scores: list[float]) -> bool:
    # ponytail: stub — define threshold when eval data is available

    if not scores:
        return True

    is_abstein = max(scores) <= RELEVANCE_THRESHOLD

    return is_abstein
