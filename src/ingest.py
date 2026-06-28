"""
ingest.py — Text extraction and cleaning from 10-K HTML reports.
Step 1 of the RAG Financial pipeline.

Input:  HTML files in data/reports/
Output: List of documents with clean text and metadata, ready for chunking.
"""

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/reports")

# Ticker → company name mapping for metadata
COMPANIES = {
    "AAPL": "Apple",
    "JPM": "JPMorgan",
    "KO": "Coca-Cola",
    "PFE": "Pfizer",
    "XOM": "ExxonMobil",
}


# ---------------------------------------------------------------------------
# Cleaning functions
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
    # Remove lines that contain only numbers (page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove typical 10-K header/footer pattern
    # Example: "Apple Inc. | 2025 Form 10-K | 12"
    text = re.sub(r".+Form 10-K.+", "", text)

    # Collapse 3+ consecutive blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def is_useful(text: str, min_words: int = 50) -> bool:
    """
    Returns True if the text has enough content to be useful for RAG.

    Filters out:
    - Cover pages and signature pages (very short)
    - Failed extractions (near-empty text)
    """
    words = text.split()
    return len(words) >= min_words


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_from_html(path: Path, ticker: str) -> dict | None:
    """
    Opens an HTML 10-K file, extracts clean text using BeautifulSoup,
    and returns a document dict with metadata.

    One HTML file = one annual report = one document.

    Returns None if the extracted text is not useful.
    """
    company = COMPANIES.get(ticker, ticker)

    print(f"  Processing {ticker} ({company}) — {path.name}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    # Parse HTML and extract visible text only (no tags, no scripts, no styles)
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, and iXBRL header (XBRL metadata, not human text)
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name == "ix:header"):
        tag.decompose()

    raw_text = soup.get_text(separator="\n", strip=True)
    text = clean_text(raw_text)

    if not is_useful(text):
        print(f"    ⚠️  Not enough content extracted from {path.name}, skipping.")
        return None

    word_count = len(text.split())
    print(f"    → {word_count:,} words extracted")

    return {
        "text": text,
        "company": company,
        "ticker": ticker,
        "source": path.name,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def load_reports() -> list[dict]:
    """
    Reads all HTML files in data/reports/ and returns a list of documents.

    Expected file naming convention:
        AAPL_10K_2025.html
        JPM_10K_2025.html
        ...
    The ticker is inferred from the first part of the filename before '_'.
    """
    html_files = sorted(DATA_DIR.glob("*.html")) + sorted(DATA_DIR.glob("*.htm"))

    if not html_files:
        raise FileNotFoundError(
            f"No HTML files found in {DATA_DIR}. "
            "Download the 10-K reports from EDGAR and save them there."
        )

    print(f"\n📂 Found {len(html_files)} HTML files in {DATA_DIR}\n")

    documents = []

    for path in html_files:
        # Infer ticker from filename (e.g. aapl-20250927.html → AAPL)
        ticker = path.stem.split("-")[0].upper()

        if ticker not in COMPANIES:
            print(f"  ⚠️  Ticker '{ticker}' not recognized, skipping {path.name}")
            continue

        doc = extract_from_html(path, ticker)
        if doc:
            documents.append(doc)

    print(f"\n✅ Total documents loaded: {len(documents)}")
    return documents


# ---------------------------------------------------------------------------
# Stats (useful for debugging and cost estimation)
# ---------------------------------------------------------------------------


def stats(documents: list[dict]) -> None:
    """Prints a summary of extracted content per company."""
    print("\n📊 Summary per company:")
    print(f"  {'Ticker':<8} {'Company':<15} {'Words':>10} {'Approx. tokens':>16}")
    print("  " + "-" * 54)

    for doc in sorted(documents, key=lambda d: d["ticker"]):
        word_count = len(doc["text"].split())
        # 1 word ≈ 1.3 tokens
        token_count = int(word_count * 1.3)
        print(
            f"  {doc['ticker']:<8} {doc['company']:<15} {word_count:>10,} {token_count:>16,}"
        )


# ---------------------------------------------------------------------------
# Direct execution — for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    documents = load_reports()
    stats(documents)

    # Show a text sample from the first document
    if documents:
        sample = documents[0]
        print(f"\n📄 Sample — {sample['company']} ({sample['source']}):")
        print("-" * 60)
        print(sample["text"][:600])
        print("...")
