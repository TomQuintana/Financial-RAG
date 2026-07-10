"""Graph node that sets the abstain flag from the retrieval scores."""

from ..agents.abstain_agent import abstain_agent
from ..graph.state import RAGState


def abstain_node(state: RAGState) -> RAGState:
    """Set ``abstain`` on the state from the current scores."""
    return {**state, "abstain": abstain_agent(state["scores"])}
