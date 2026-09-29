from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import PatientProfile, TrialDetailResponse, TrialSearchRequest, TrialSearchResponse
from backend.app.services.clinicaltrials import ClinicalTrialsClient
from backend.app.services.funnel import build_funnel
from backend.app.services.recommendations import RecommendationService

router = APIRouter(prefix="/trials", tags=["trials"])
client = ClinicalTrialsClient()
recommendations = RecommendationService()


@router.post("/search", response_model=TrialSearchResponse)
async def search_trials(request: TrialSearchRequest) -> TrialSearchResponse:
    trials, source = await client.search(request)
    profile = PatientProfile(
        condition=request.condition or request.query,
        location=request.location,
        phase_preferences=[request.phase] if request.phase else [],
        notes=request.query,
    )
    # See recommendations.py for why this ranks the full set before truncating.
    ranked = recommendations.recommend(profile, trials, limit=len(trials))
    results = ranked[:request.page_size]
    funnel = build_funnel(trials, ranked, len(results))
    return TrialSearchResponse(query=request.query, total=len(results), results=results, source=source, funnel=funnel)


@router.get("/{nct_id}", response_model=TrialDetailResponse)
async def trial_detail(nct_id: str) -> TrialDetailResponse:
    trial, source = await client.get_trial(nct_id)
    if trial is None:
        raise HTTPException(status_code=404, detail="Trial not found")
    return TrialDetailResponse(trial=trial, source=source)
