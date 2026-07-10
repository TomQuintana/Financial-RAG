"""Graph node that generates the answer from the query and documents."""

from ..agents.generate_output_agent import generate_output_agent
from ..graph.state import RAGState


def generate_node(state: RAGState) -> RAGState:
    """Set ``answer`` on the state from the query and retrieved documents."""
    return {**state, "answer": generate_output_agent(state["query"], state["documents"])}
