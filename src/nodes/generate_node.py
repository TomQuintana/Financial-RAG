from ..agents.generate_output_agent import generate_output_agent
from ..graph.state import RAGState


def generate_node(state: RAGState) -> RAGState:
    return {**state, "answer": generate_output_agent(state["query"], state["documents"])}
