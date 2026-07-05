# Financial RAG — Claude Rules

## API Keys
- Never read, print, or log `.env` contents
- Never echo or expose `OPENAI_API_KEY` or any `*_KEY` / `*_SECRET` variable
- Keys load via `load_dotenv()` at module level in each file — trust that

## Project Structure
```
src/
  ingest.py    — Steps 1-3: extract HTML → chunk → embed into ChromaDB
  retrieve.py  — Step 4: similarity search against ChromaDB
  generate.py  — Step 5: LLM answer generation with retrieved context
  cache.py     — Query/response caching layer
  limiter.py   — API rate limiting (slowapi)
  app.py       — FastAPI app, wires everything together
eval/
  run_eval.py  — RAGAS evaluation pipeline
  eval_questions.json — ground-truth Q&A pairs for eval
data/
  reports/     — Raw 10-K HTML files from EDGAR (gitignored)
  chroma_db/   — Persisted ChromaDB vectors (gitignored)
```

## Testing
- Eval framework: RAGAS (`ragas` package already installed)
- Run evals with: `uv run python eval/run_eval.py`
- Metrics: faithfulness, answer relevancy, context precision, context recall
- Ground-truth questions in `eval/eval_questions.json`
- Never modify eval questions to make scores look better

## Caching (`src/cache.py`)
- Cache layer sits between API and retrieval/generation
- Key: hash of the query string
- Do not cache errors or empty responses
- TTL and backend TBD — check `cache.py` before assuming implementation

## Rate Limiting (`src/limiter.py`)
- Uses `slowapi` (already installed)
- Applied at FastAPI route level in `app.py`
- Do not bypass or disable limits in tests — use lower limits instead

## General
- Run from project root, not from `src/`: paths like `data/reports/` are relative to root
- ChromaDB collection name: `financial_reports`
- Embedding model: `text-embedding-3-small`
- LLM: check `generate.py` before assuming model name
