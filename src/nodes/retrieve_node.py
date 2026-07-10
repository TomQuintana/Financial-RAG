"""Graph node that retrieves documents for the query."""

from ..agents.retrieve_agent import retrieve_agent
from ..graph.state import RAGState


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve documents and update ``documents`` and ``scores`` on the state."""
    docs, scores = retrieve_agent(state["query"])
    return {**state, "documents": docs, "scores": scores}
