"""FastAPI app exposing the RAG pipeline over an HTTP endpoint."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph.graph_service import graph_service

app = FastAPI(title="Financial RAG")


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str


@app.post("/query")
def query(req: QueryRequest):
    """Answer a query with the RAG pipeline, or raise 500 on pipeline error."""
    result = graph_service.process_query(req.query)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
