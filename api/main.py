"""FastAPI app exposing the RAG pipeline over an HTTP endpoint."""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.graph.graph_service import graph_service
from src.limiter import limiter
from src.settings import settings

app = FastAPI(title="Financial RAG")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str


@app.post("/query")
@limiter.limit(settings.RATE_LIMIT)
def query(request: Request, req: QueryRequest):
    """Answer a query with the RAG pipeline, or raise 500 on pipeline error."""
    result = graph_service.process_query(req.query)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
