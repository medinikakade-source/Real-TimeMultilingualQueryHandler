# backend/llm_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.simple_retriever import build_index, query_index
from backend.llm_stub import LLMStub
from backend.translator import detect_language, translate_text

router = APIRouter()
_llm = LLMStub()

# Build in-memory index once at import time
_DOCS, _TEXTS, _VECTORS = build_index()

class AnswerRequest(BaseModel):
    query: str
    k: int = 4
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # informational for embeddings

class SourceItem(BaseModel):
    source: str
    chunk: str
    score: float

class AnswerResponse(BaseModel):
    query: str
    detected_language: str
    query_en: str
    answer_en: str
    answer_local: str
    sources: List[SourceItem]
    llm_metadata: dict

import re

def sentence_case(text: str) -> str:
    """
    Convert text into basic sentence case: lowercase everything,
    then capitalize the first letter of each sentence.
    Not perfect linguistically but good for demo outputs.
    """
    if not text:
        return text
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # split into sentences by [.?!] followed by space (keeps the punctuation)
    parts = re.split(r'([.?!])\s+', text)
    if len(parts) == 1:
        return text.capitalize()
    out = []
    # parts is like [sent0, punct0, sent1, punct1, ...] sometimes
    i = 0
    while i < len(parts):
        sent = parts[i].strip()
        punct = ""
        if i + 1 < len(parts) and re.match(r'[.?!]', parts[i+1]):
            punct = parts[i+1]
            i += 2
        else:
            i += 1
        if sent:
            s = sent.lower()
            s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            out.append(s + (punct + " " if punct else ""))
    return "".join(out).strip()


def _clean_text_for_prompt(text: str) -> str:
    """
    Remove markdown headings and trim noisy whitespace to keep prompts and returned sources tidy.
    """
    if not text:
        return text
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove lines that are markdown headings (start with #)
    lines = [ln for ln in text.split("\n") if not re.match(r'^\s*#{1,6}\s+', ln)]
    cleaned = "\n".join(lines).strip()
    # Collapse multiple blank lines to a single blank line
    cleaned = re.sub(r'\n{2,}', '\n\n', cleaned)
    return cleaned

def _compose_prompt_en(query_en: str, retrieved):
    """
    Build an English prompt for the LLM. Each retrieved chunk is translated to English,
    cleaned, and included. The prompt includes an exact 'Question:' line so the stub parses it.
    """
    instruction = (
        "You are a helpful assistant that answers using ONLY the provided CONTEXT. "
        "If the context doesn't contain enough information, say you don't know.\n\n"
    )
    context_text = "CONTEXT:\n"
    for score, doc in retrieved:
        # translate chunk to English explicitly for the prompt
        doc_en = translate_text(doc["text"], target="en")
        # clean the translated chunk (remove Markdown headings, extra whitespace)
        doc_en = _clean_text_for_prompt(doc_en)
        header = f"source: {doc['source']} | chunk: {doc['chunk']} | score: {score:.4f}"
        context_text += f"--- {header}\n{doc_en}\n\n"
    prompt = f"{instruction}{context_text}\nQuestion: {query_en}\n"
    return prompt




@router.post("/answer_translate", response_model=AnswerResponse)
async def answer_translate(req: AnswerRequest):
    # Validate input
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    # 1) Detect user language
    user_lang = detect_language(req.query)

    # 2) Translate user query to English
    query_en = req.query if user_lang == "en" else translate_text(req.query, target="en")

    # 3) Retrieve using English query (we translated it)
    try:
        results = query_index(query_en, _DOCS, _VECTORS, model_name=req.model_name, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # 4) Compose English prompt (contexts translated into English)
    prompt = _compose_prompt_en(query_en, results)

    # 5) Call the (stub) LLM to generate an English answer
    llm_out = _llm.generate(prompt)
    answer_en = llm_out.get("answer", "")
    answer_en = sentence_case(answer_en)


    # 6) Translate the English answer back to user's language (if needed)
    answer_local = answer_en if user_lang == "en" else translate_text(answer_en, target=user_lang)

    # 7) Build sources list
    sources = []
    for score, doc in results:
    # include a cleaned chunk in the API response (translate to English and clean, but keep original as well if you want)
        cleaned_chunk = _clean_text_for_prompt(translate_text(doc["text"], target="en"))
        cleaned_chunk = sentence_case(cleaned_chunk)
        sources.append(SourceItem(source=doc["source"], chunk=cleaned_chunk, score=float(score)))


    return AnswerResponse(
        query=req.query,
        detected_language=user_lang,
        query_en=query_en,
        answer_en=answer_en,
        answer_local=answer_local,
        sources=sources,
        llm_metadata=llm_out.get("metadata", {}),
    )
