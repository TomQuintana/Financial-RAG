from ..agents.retrieve_agent import retrieve_agent
from ..graph.state import RAGState


def retrieve_node(state: RAGState) -> RAGState:
    docs, scores = retrieve_agent(state["query"])
    return {**state, "documents": docs, "scores": scores}
