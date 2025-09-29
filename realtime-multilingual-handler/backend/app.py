# backend/app.py
from fastapi import FastAPI

app = FastAPI(title="Realtime Multilingual Query Handler - Backend")

@app.get("/ping")
async def ping():
    """
    Simple healthcheck / smoke test endpoint.
    Returns a tiny JSON payload so we can test the server easily.
    """
    return {"status": "ok", "message": "pong"}
