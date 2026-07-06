"""
ingest.py — Text extraction, cleaning, chunking and indexing from 10-K HTML reports.
Steps 1, 2 and 3 of the RAG Financial pipeline.

Input:  HTML files in data/reports/
Output: ChromaDB collection with embedded chunks, ready for retrieval.
"""

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from ..helpers.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/reports")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION = "financial_reports"

COMPANIES = {
    "AAPL": "Apple",
    "JPM": "JPMorgan",
    "KO": "Coca-Cola",
    "PFE": "Pfizer",
    "XOM": "ExxonMobil",
}

# ~800 tokens ≈ 3200 characters (1 token ≈ 4 chars for English financial text)
CHUNK_SIZE = 3200
CHUNK_OVERLAP = 400  # ~100 tokens overlap

# text-embedding-3-small: best balance of quality and cost (~$0.02/1M tokens)
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from a 10-K HTML file.

    Common problems in 10-K reports:
    - Lines containing only page numbers
    - Repeated headers/footers like "Apple Inc. | Form 10-K | 7"
    - Excessive blank lines between paragraphs
    - Leading/trailing whitespace on each line
    """
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r".+Form 10-K.+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def is_useful(text: str, min_words: int = 50) -> bool:
    return len(text.split()) >= min_words


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_from_html(path: Path, ticker: str) -> dict | None:
    company = COMPANIES.get(ticker, ticker)
    logger.info("Processing %s (%s) — %s", ticker, company, path.name)

    with open(path, encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, and iXBRL header (XBRL metadata, not human text)
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name == "ix:header"):
        tag.decompose()

    text = clean_text(soup.get_text(separator="\n", strip=True))

    if not is_useful(text):
        logger.warning("Not enough content in %s, skipping.", path.name)
        return None

    logger.info("→ %s words extracted", f"{len(text.split()):,}")
    return {"text": text, "company": company, "ticker": ticker, "source": path.name}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Splits each document into chunks of ~800 tokens with ~100 token overlap.

    Separators tried in order: paragraph → line → sentence → word → char.
    Each chunk inherits parent metadata plus chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        raw_chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(raw_chunks):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "company": doc["company"],
                    "ticker": doc["ticker"],
                    "source": doc["source"],
                    "chunk_index": i,
                }
            )
        logger.info("%s → %d chunks", doc["ticker"], len(raw_chunks))

    logger.info("✅ Total chunks: %d", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding + Indexing
# ---------------------------------------------------------------------------


def build_embeddings(chunks: list[dict]) -> Chroma:
    """
    Embeds all chunks with OpenAI text-embedding-3-small and stores them
    in a persistent ChromaDB collection at CHROMA_DIR.

    IDs are ticker + chunk_index (e.g. "AAPL_0042") to allow safe re-runs
    without duplicating existing chunks.
    """
    logger.info("🔢 Loading embedding model: %s", EMBEDDING_MODEL)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "company": chunk["company"],
            "ticker": chunk["ticker"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]
    ids = [f"{chunk['ticker']}_{chunk['chunk_index']:04d}" for chunk in chunks]

    logger.info("📥 Indexing %d chunks into ChromaDB at %s...", len(texts), CHROMA_DIR)

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )

    logger.info("✅ Index built — %d chunks stored in '%s'", len(texts), COLLECTION)
    return vector_store


def load_reports() -> list[dict]:
    html_files = sorted(DATA_DIR.glob("*.html")) + sorted(DATA_DIR.glob("*.htm"))

    if not html_files:
        raise FileNotFoundError(
            f"No HTML files found in {DATA_DIR}. "
            "Download 10-K reports from EDGAR and save them there."
        )

    logger.info("📂 Found %d HTML files in %s", len(html_files), DATA_DIR)
    documents = []

    for path in html_files:
        # EDGAR naming: aapl-20250927.html → AAPL
        ticker = path.stem.split("-")[0].upper()
        if ticker not in COMPANIES:
            logger.warning("Ticker '%s' not recognized, skipping %s", ticker, path.name)
            continue
        doc = extract_from_html(path, ticker)
        if doc:
            documents.append(doc)

    logger.info("✅ Total documents loaded: %d", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def stats(chunks: list[dict]) -> None:
    logger.info("📊 Chunk summary per company:")
    logger.info(
        "  %-8s %-15s %8s %18s", "Ticker", "Company", "Chunks", "Avg. chars/chunk"
    )
    logger.info("  %s", "-" * 54)

    per_ticker: dict[str, list] = {}
    for chunk in chunks:
        per_ticker.setdefault(chunk["ticker"], []).append(chunk)

    for ticker, ticker_chunks in sorted(per_ticker.items()):
        company = ticker_chunks[0]["company"]
        avg_chars = int(sum(len(c["text"]) for c in ticker_chunks) / len(ticker_chunks))
        logger.info(
            "  %-8s %-15s %8d %18s",
            ticker,
            company,
            len(ticker_chunks),
            f"{avg_chars:,}",
        )


# ---------------------------------------------------------------------------
# Direct execution — runs the full ingestion pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1 — extract text
    documents = load_reports()

    # Step 2 — chunk
    logger.info("✂️  Chunking documents...")
    chunks = chunk_documents(documents)
    stats(chunks)

    # Step 3 — embed + index
    build_embeddings(chunks)
