from src.agents.abstain_agent import RELEVANCE_THRESHOLD, abstain_agent


def test_abstains_when_no_scores():
    """Verifies the agent abstains when the scores list is empty.

    Without retrieved documents there is nothing to evaluate,
    so the pipeline must stop before calling the LLM.
    """
    assert abstain_agent([]) is True


def test_abstains_when_all_scores_are_negative():
    """Verifies the agent abstains when all CrossEncoder logits are negative.

    Negative logits indicate that none of the retrieved documents
    are relevant to the query.
    """
    assert abstain_agent([-8.0, -5.0, -3.0]) is True


def test_does_not_abstain_when_one_score_is_high():
    """Verifies the agent does not abstain if at least one score is high.

    A single relevant document (high logit) is enough to proceed to
    the LLM. The decision is based on the maximum score, not the average.
    """
    assert abstain_agent([9.0, -2.0, -5.0]) is False


def test_abstains_at_exact_threshold_boundary():
    """Verifies that a score exactly at the threshold causes abstention.

    The condition uses <= (not <), so the boundary value also abstains.
    Conservative decision: when in doubt, do not generate a response.
    """
    assert abstain_agent([RELEVANCE_THRESHOLD]) is True


def test_does_not_abstain_when_score_is_above_threshold():
    """Verifies that a score above the threshold allows the pipeline to continue.

    A logit of 0.1 exceeds the threshold of 0.0, indicating enough
    relevance to attempt generating a response.
    """
    assert abstain_agent([0.1]) is False
