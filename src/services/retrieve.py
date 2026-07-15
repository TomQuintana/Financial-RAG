"""retrieve.py — Similarity search against the ChromaDB collection.

Step 4 of the RAG Financial pipeline.

Input:  Query string
Output: List of LangChain Documents with metadata (company, ticker, chunk_index)
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..helpers.logger import get_logger

# TODO: pass to other file
from .ingest import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL

load_dotenv()

logger = get_logger(__name__)


def load_vectorstore() -> Chroma:
    """Connect to persisted ChromaDB — does not re-index."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )
