from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import json
import logging

from backend.app.models.schemas import PatientProfile, TrialSearchRequest, TrialSearchResponse
from backend.app.services.clinicaltrials import ClinicalTrialsClient
from backend.app.services.funnel import build_funnel
from backend.app.services.recommendations import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
client = ClinicalTrialsClient()
recommendations = RecommendationService()

RESULT_LIMIT = 10


@router.post("/stream")
async def stream_recommendations(profile: PatientProfile):
    """Progress at real operation boundaries; the existing scoring path is unchanged."""
    async def events():
        try:
            search_request = TrialSearchRequest(
                query=profile.condition, condition=profile.condition,
                location=profile.location, page_size=25,
            )
            trials, source = await client.search(search_request)
            yield json.dumps({"completed": 1}) + "\n"
            ranked = await run_in_threadpool(recommendations.recommend, profile, trials, limit=len(trials))
            yield json.dumps({"completed": 2}) + "\n"
            results = ranked[:RESULT_LIMIT]
            response = TrialSearchResponse(
                query=profile.condition, total=len(results), results=results, source=source,
                funnel=build_funnel(trials, ranked, len(results)),
            )
            yield json.dumps({"completed": 3, "response": response.model_dump(mode="json")}) + "\n"
        except Exception:
            logging.getLogger(__name__).exception("Streaming recommendation request failed")
            yield json.dumps({"error": "Search could not be completed"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
