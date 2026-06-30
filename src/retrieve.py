"""
retrieve.py — Similarity search against the ChromaDB collection.
Step 4 of the RAG Financial pipeline.

Input:  Query string
Output: List of LangChain Documents with metadata (company, ticker, chunk_index)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ingest import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL

load_dotenv()


def load_vectorstore() -> Chroma:
    """Connect to persisted ChromaDB — does not re-index."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(query: str, k: int = 5, ticker: str | None = None) -> list[Document]:
    """Similarity search. Filter by ticker if provided."""
    vs = load_vectorstore()
    where = {"ticker": ticker} if ticker else None
    return vs.similarity_search(query, k=k, filter=where)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What was the total revenue of APPL?"
    results = retrieve(query, k=3)
    for i, doc in enumerate(results):
        print(
            f"\nResult {i + 1} — {doc.metadata['company']} (chunk {doc.metadata['chunk_index']}):"
        )
        print(f"  {doc.page_content[:200]}...")
