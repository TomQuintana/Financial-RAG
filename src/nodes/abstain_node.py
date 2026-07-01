from ..agents.abstain_agent import abstain_agent
from ..graph.state import RAGState


def abstain_node(state: RAGState) -> RAGState:
    return {**state, "abstain": abstain_agent(state["scores"])}
