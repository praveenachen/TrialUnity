import json

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.routes import recommendations as route
from backend.app.models.schemas import Trial


def test_stream_matches_existing_response_and_reports_real_boundaries(monkeypatch):
    async def search(request):
        return [Trial(nct_id="NCT1", title="Cancer study", status="RECRUITING", conditions=["Cancer"])], "sample-data"

    monkeypatch.setattr(route.client, "search", search)
    with TestClient(app) as client:
        expected = client.post("/api/recommendations", json={"condition": "cancer"}).json()
        response = client.post("/api/recommendations/stream", json={"condition": "cancer"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["completed"] for event in events] == [1, 2, 3]
    assert events[-1]["response"] == expected


def test_search_failure_does_not_check_off_work(monkeypatch):
    async def search(request):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(route.client, "search", search)
    with TestClient(app) as client:
        response = client.post("/api/recommendations/stream", json={"condition": "cancer"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert len(events) == 1
    assert "error" in events[0]
    assert "completed" not in events[0]


def test_ranking_failure_keeps_only_retrieval_completed(monkeypatch):
    async def search(request):
        return [], "sample-data"

    def rank(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(route.client, "search", search)
    monkeypatch.setattr(route.recommendations, "recommend", rank)
    with TestClient(app) as client:
        response = client.post("/api/recommendations/stream", json={"condition": "cancer"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {"completed": 1}
    assert len(events) == 2
    assert "error" in events[1]
