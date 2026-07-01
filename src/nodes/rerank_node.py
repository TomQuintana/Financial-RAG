from ..agents.rerank_agent import rerank_agent
from ..graph.state import RAGState


def rerank_node(state: RAGState) -> RAGState:
    docs, scores = rerank_agent(state["documents"], state["scores"])
    return {**state, "documents": docs, "scores": scores}
