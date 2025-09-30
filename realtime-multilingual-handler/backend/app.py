# backend/app.py
from fastapi import FastAPI
from backend.simple_api import router as simple_router

app = FastAPI(title="Realtime Multilingual Query Handler - Backend")

# include the simple retriever routes
app.include_router(simple_router, prefix="/api")

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
