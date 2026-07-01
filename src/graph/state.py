from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    query: str
    documents: list[Document]
    scores: list[float]
    answer: str
    abstain: bool
