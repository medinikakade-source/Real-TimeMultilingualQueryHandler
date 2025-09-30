# backend/llm_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from backend.simple_retriever import build_index, query_index
from backend.llm_stub import LLMStub

router = APIRouter()
_llm = LLMStub()

# Build in-memory index once (reused while the server runs)
_DOCS, _TEXTS, _VECTORS = build_index()

class AnswerRequest(BaseModel):
    query: str
    k: int = 4
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # informational only for stub

class SourceItem(BaseModel):
    source: str
    chunk: str
    score: float

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    llm_metadata: dict

def _compose_prompt(query: str, retrieved: List[tuple]) -> str:
    """
    Compose a simple prompt template:
    - Instruction + contexts (top-k)
    - A clear `Question:` line so the stub can parse it
    """
    instruction = (
        "You are a helpful assistant that answers using ONLY the provided CONTEXT. "
        "If the context doesn't contain enough information, say you don't know.\n\n"
    )
    context_text = "CONTEXT:\n"
    for idx, (score, doc) in enumerate(retrieved):
        # doc is dict with 'text' and 'source' (as simple_retriever returns)
        header = f"source: {doc['source']} | chunk: {doc['chunk']} | score: {score:.4f}"
        context_text += f"--- {header}\n{doc['text']}\n\n"
    prompt = f"{instruction}{context_text}\nQuestion: {query}\n"
    return prompt

@router.post("/answer", response_model=AnswerResponse)
async def answer_endpoint(req: AnswerRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")
    # get top-k from in-memory index
    results = query_index(req.query, _DOCS, _VECTORS, model_name=req.model_name, k=req.k)
    # results is list of (score, doc)
    # Compose prompt with retrieved contexts
    prompt = _compose_prompt(req.query, results)
    # Call stub LLM
    llm_out = _llm.generate(prompt)
    # Map results to response sources list
    sources = []
    for score, doc in results:
        sources.append(SourceItem(source=doc["source"], chunk=doc["text"], score=float(score)))
    return AnswerResponse(
        query=req.query,
        answer=llm_out["answer"],
        sources=sources,
        llm_metadata=llm_out["metadata"],
    )
