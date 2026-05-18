from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import assistant, health, recommendations, trials
from backend.app.core.config import get_settings


settings = get_settings()
app = FastAPI(
    title="TrialUnity API",
    description="Explainable clinical trial retrieval and recommendation services.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(trials.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(assistant.router, prefix="/api")

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
