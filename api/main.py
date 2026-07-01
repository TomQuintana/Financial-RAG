from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph.graph_service import graph_service

app = FastAPI(title="Financial RAG")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(req: QueryRequest):
    result = graph_service.process_query(req.query)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
