# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.simple_api import router as simple_router
from backend.llm_api import router as llm_trans_router

app = FastAPI(title="Realtime Multilingual Query Handler - Backend")

# Allow CORS for local dev (adjust origins for production)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(simple_router, prefix="/api")
app.include_router(llm_trans_router, prefix="/api")

# Serve a static single-page UI from / (files placed in backend/static/)
app.mount("/", StaticFiles(directory="backend/static", html=True), name="static")

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
