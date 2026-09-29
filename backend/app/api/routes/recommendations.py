from fastapi import APIRouter

from backend.app.models.schemas import PatientProfile, TrialSearchRequest, TrialSearchResponse
from backend.app.services.clinicaltrials import ClinicalTrialsClient
from backend.app.services.funnel import build_funnel
from backend.app.services.recommendations import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
client = ClinicalTrialsClient()
recommendations = RecommendationService()

RESULT_LIMIT = 10


@router.post("", response_model=TrialSearchResponse)
async def recommend_trials(profile: PatientProfile) -> TrialSearchResponse:
    search_request = TrialSearchRequest(
        query=profile.condition,
        condition=profile.condition,
        location=profile.location,
        page_size=25,
    )
    trials, source = await client.search(search_request)
    # Ranked over every candidate first (identical scores/order to limit=10 -- `limit`
    # only truncates the final list) so the funnel can truthfully count the full set
    # before slicing, without ranking anything twice.
    ranked = recommendations.recommend(profile, trials, limit=len(trials))
    results = ranked[:RESULT_LIMIT]
    funnel = build_funnel(trials, ranked, len(results))
    return TrialSearchResponse(query=profile.condition, total=len(results), results=results, source=source, funnel=funnel)
