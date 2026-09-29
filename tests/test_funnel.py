from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.schemas import PatientProfile, Trial, TrialRecommendation
from backend.app.services.funnel import build_funnel
from backend.app.services.recommendations import RecommendationService


def trial(**changes):
    return Trial(**(dict(
        nct_id="NCT1", title="Cancer study", status="RECRUITING", conditions=["Cancer"],
    ) | changes))


def test_build_funnel_counts_are_truthful_and_independent_of_scoring() -> None:
    trials = [
        trial(nct_id="1", status="RECRUITING"),
        trial(nct_id="2", status="RECRUITING", minimum_age="99 Years"),  # will be incompatible for most ages
        trial(nct_id="3", status="COMPLETED"),
    ]
    service = RecommendationService()
    patient = PatientProfile(condition="cancer", age=40)
    ranked = service.recommend(patient, trials, limit=len(trials))

    funnel = build_funnel(trials, ranked, returned=2)

    assert funnel.candidate_trials == 3
    assert funnel.recruiting_trials == 2
    assert funnel.ranked_matches == 2
    # trial "2" (age 99 min) is incompatible for a 40-year-old; trial "3" is
    # incompatible via recruitment status; trial "1" has no disqualifying criteria.
    assert funnel.structured_eligible_trials == 1


def test_build_funnel_handles_no_candidates() -> None:
    funnel = build_funnel([], [], returned=0)
    assert funnel.candidate_trials == 0
    assert funnel.recruiting_trials == 0
    assert funnel.structured_eligible_trials == 0
    assert funnel.ranked_matches == 0


def test_recommendations_endpoint_includes_a_consistent_funnel(monkeypatch) -> None:
    from backend.app.api.routes import recommendations as route

    trials = [trial(nct_id="1"), trial(nct_id="2", status="COMPLETED")]

    async def fake_search(request):
        return trials, "sample-data"

    monkeypatch.setattr(route.client, "search", fake_search)
    with TestClient(app) as client:
        response = client.post("/api/recommendations", json={"condition": "cancer"})
    assert response.status_code == 200
    body = response.json()
    funnel = body["funnel"]
    assert funnel["candidate_trials"] == 2
    assert funnel["recruiting_trials"] == 1
    assert funnel["ranked_matches"] == len(body["results"])
    assert funnel["ranked_matches"] <= funnel["candidate_trials"]


def test_search_results_are_identical_whether_or_not_the_funnel_is_computed() -> None:
    # Guards against the funnel computation accidentally changing what gets ranked.
    trials = [trial(nct_id=str(i)) for i in range(5)]
    patient = PatientProfile(condition="cancer")
    service = RecommendationService()

    direct = service.recommend(patient, trials, limit=3)
    full_then_sliced = service.recommend(patient, trials, limit=len(trials))[:3]

    assert [r.trial.nct_id for r in direct] == [r.trial.nct_id for r in full_then_sliced]
    assert [r.score for r in direct] == [r.score for r in full_then_sliced]
