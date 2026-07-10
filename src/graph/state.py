"""Shared state passed between nodes of the RAG graph."""

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    """State carried through the RAG graph across all nodes."""

    query: str
    documents: list[Document]
    scores: list[float]
    answer: str
    abstain: bool
