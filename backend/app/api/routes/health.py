import os

from fastapi import APIRouter

from backend.app.core.config import get_settings
from backend.app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.app_env,
        llm_enabled=settings.enable_llm,
        openai_key_configured=bool(os.getenv("OPENAI_API_KEY") or settings.openai_api_key),
    )
