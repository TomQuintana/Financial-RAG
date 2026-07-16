"""One-off script: measure real cosine distances to pick CACHE_DISTANCE_THRESHOLD.

Run with: uv run python eval/measure_cache_threshold.py
"""

import sys
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.services.ingest import EMBEDDING_MODEL

# (original, paraphrase that should HIT, confusable that should MISS)
PAIRS = [
    (
        "What was Apple's total revenue (net sales) in fiscal year 2025, "
        "and how did it compare to the prior year?",
        "How much revenue did Apple make in FY2025 versus the year before?",
        "What was Apple's total revenue in fiscal year 2023?",
    ),
    (
        "How much did Apple spend on research and development (R&D) during fiscal year 2025?",
        "What was Apple's R&D spend in FY2025?",
        "How much did Apple spend on marketing during fiscal year 2025?",
    ),
    (
        "What share repurchase programs did Apple authorize or execute during fiscal year 2025?",
        "Which stock buyback programs did Apple run in FY2025?",
        "What share repurchase programs did Apple authorize during fiscal year 2022?",
    ),
    (
        "What are the main competition-related risk factors Apple discloses in its 10-K?",
        "What competitive risks does Apple's 10-K describe?",
        "What are the main cybersecurity-related risk factors Apple discloses in its 10-K?",
    ),
]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity, matching Chroma's hnsw:space=cosine distance."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return 1 - dot / (norm_a * norm_b)


def main() -> None:
    """Print cosine distances for each HIT/MISS pair in PAIRS."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print(f"{'kind':<6} {'distance':>8}  original / variant")
    for original, paraphrase, confusable in PAIRS:
        orig_vec = embeddings.embed_query(original)
        for variant, kind in [(paraphrase, "HIT"), (confusable, "MISS")]:
            dist = cosine_distance(orig_vec, embeddings.embed_query(variant))
            print(f"{kind:<6} {dist:>8.4f}  {original[:60]!r} / {variant[:60]!r}")


if __name__ == "__main__":
    main()
