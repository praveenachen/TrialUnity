from fastapi import APIRouter

from backend.app.models.schemas import PatientProfile, TrialSearchRequest, TrialSearchResponse
from backend.app.services.clinicaltrials import ClinicalTrialsClient
from backend.app.services.recommendations import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
client = ClinicalTrialsClient()
recommendations = RecommendationService()


@router.post("", response_model=TrialSearchResponse)
async def recommend_trials(profile: PatientProfile) -> TrialSearchResponse:
    search_request = TrialSearchRequest(
        query=" ".join([profile.condition, profile.notes or ""]).strip(),
        condition=profile.condition,
        location=profile.location,
        page_size=25,
    )
    trials, source = await client.search(search_request)
    ranked = recommendations.recommend(profile, trials, limit=10)
    return TrialSearchResponse(query=profile.condition, total=len(ranked), results=ranked, source=source)
