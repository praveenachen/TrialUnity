from fastapi import APIRouter

from backend.app.models.schemas import AssistantRequest, AssistantResponse
from backend.app.services.llm import TrialAssistant

router = APIRouter(prefix="/assistant", tags=["assistant"])
assistant = TrialAssistant()


@router.post("/answer", response_model=AssistantResponse)
async def answer_question(request: AssistantRequest) -> AssistantResponse:
    return await assistant.answer(request)
