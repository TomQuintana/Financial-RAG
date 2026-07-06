"""
generate.py — LangGraph RAG pipeline.
Steps 4-5: retrieve → rerank → abstain → generate.
"""

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import load_vectorstore

from ..helpers.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class RAGState(TypedDict):
    query: str
    documents: list[Document]
    scores: list[float]
    answer: str
    abstain: bool


def retrieve_node(state: RAGState) -> RAGState:
    vs = load_vectorstore()
    results = vs.similarity_search_with_score(state["query"], k=5)
    docs, scores = zip(*results) if results else ([], [])
    return {**state, "documents": list(docs), "scores": list(scores)}


def rerank_node(state: RAGState) -> RAGState:
    # ponytail: stub — add cross-encoder (e.g. ms-marco) when retrieval quality needs boost
    return state


def abstain_node(state: RAGState) -> RAGState:
    # ponytail: stub — set threshold when eval data is available
    return {**state, "abstain": False}


def generate_node(state: RAGState) -> RAGState:
    context = "\n\n".join(d.page_content for d in state["documents"])
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}"
    )
    answer = llm.invoke(prompt).content
    return {**state, "answer": answer}  # type: ignore


def route_after_abstain(state: RAGState) -> str:
    return END if state["abstain"] else "generate"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("abstain", abstain_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "abstain")
graph.add_conditional_edges(
    "abstain", route_after_abstain, {"generate": "generate", END: END}
)
graph.add_edge("generate", END)

pipeline = graph.compile()


def run_pipeline(query: str) -> RAGState:
    return pipeline.invoke(
        {"query": query, "documents": [], "scores": [], "answer": "", "abstain": False}
    )  # type: ignore


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What was Apple's total revenue?"
    result = run_pipeline(query)
    logger.info("Answer: %s", result["answer"])
