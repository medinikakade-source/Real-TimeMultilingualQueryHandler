# backend/app.py
from fastapi import FastAPI
from backend.simple_api import router as simple_router
from backend.llm_api import router as llm_router

app = FastAPI(title="Realtime Multilingual Query Handler - Backend")

app.include_router(simple_router, prefix="/api")
app.include_router(llm_router, prefix="/api")

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
