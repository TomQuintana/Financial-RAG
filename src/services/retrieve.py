"""
retrieve.py — Similarity search against the ChromaDB collection.
Step 4 of the RAG Financial pipeline.

Input:  Query string
Output: List of LangChain Documents with metadata (company, ticker, chunk_index)
"""

import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# TODO: pass to other file
from .ingest import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL
from .logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def load_vectorstore() -> Chroma:
    """Connect to persisted ChromaDB — does not re-index."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(query: str, k: int = 5, ticker: str | None = None) -> list[Document]:
    """Similarity search. Filter by ticker if provided."""
    vectors = load_vectorstore()
    where = {"ticker": ticker} if ticker else None
    return vectors.similarity_search(query, k=k, filter=where)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What was the total revenue of APPL?"
    results = retrieve(query, k=3)
    logger.debug("Vectors results: %s", results)

    for i, doc in enumerate(results):
        logger.debug(
            "Result %d — %s (chunk %s): %s...",
            i + 1,
            doc.metadata["company"],
            doc.metadata["chunk_index"],
            doc.page_content[:200],
        )
