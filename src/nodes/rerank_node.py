"""Graph node that reranks the retrieved documents."""

from ..agents.rerank_agent import rerank_agent
from ..graph.state import RAGState


def rerank_node(state: RAGState) -> RAGState:
    """Rerank the documents and update ``documents`` and ``scores`` on the state."""
    docs, scores = rerank_agent(state["query"], state["documents"])
    return {**state, "documents": docs, "scores": scores}
