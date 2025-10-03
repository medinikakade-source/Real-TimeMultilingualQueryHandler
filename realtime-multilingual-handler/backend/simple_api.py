# backend/simple_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from backend.simple_retriever import build_index, query_index

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # dev default

class RetrievedItem(BaseModel):
    score: float
    source: str
    chunk: str

class QueryResponse(BaseModel):
    query: str
    results: List[RetrievedItem]

# Build index once at import time so we reuse embeddings in memory during dev.
# For larger datasets you'd persist/reuse vectors; here simple in-memory index is fine.
_docs, _texts, _vectors = build_index()  # uses default data/ and default model

@router.post("/simple/query", response_model=QueryResponse)
async def simple_query(req: QueryRequest):
    if not req.query or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query text is required.")
    # If you want to re-index with a different model at runtime, you'd call build_index(model_name=...)
    # Here we use the already-built in-memory vectors.
    results = query_index(req.query, _docs, _vectors, model_name=req.model_name, k=req.k)
    out = []
    for score, doc in results:
        out.append(RetrievedItem(score=float(score), source=doc["source"], chunk=doc["text"]))
    return QueryResponse(query=req.query, results=out)
