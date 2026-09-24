import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.schemas import AssistantRequest, Trial, TrialSearchRequest
from backend.app.services import clinicaltrials, llm


def mock_http(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(clinicaltrials.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))


@pytest.mark.parametrize("phase", ["PHASE2", "phase 2"])
def test_phase_query(monkeypatch, phase):
    def handler(request):
        assert request.url.params["filter.advanced"] == "AREA[Phase]PHASE2"
        assert request.url.params["query.term"] == "cancer"
        assert request.url.params["filter.overallStatus"] == "RECRUITING"
        return httpx.Response(200, json={"studies": []})
    mock_http(monkeypatch, handler)
    assert asyncio.run(clinicaltrials.ClinicalTrialsClient().search(
        TrialSearchRequest(query="cancer", phase=phase))) == ([], "clinicaltrials.gov")


def test_invalid_phase_rejected():
    with TestClient(app) as client:
        assert client.post("/api/trials/search", json={"query": "cancer", "phase": "PHASE2 OR other"}).status_code == 422


@pytest.mark.parametrize("failure", ["timeout", "http", "json", "shape"])
def test_registry_fallback(monkeypatch, caplog, failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("test")
        if failure == "http":
            return httpx.Response(503)
        if failure == "json":
            return httpx.Response(200, text="not json")
        return httpx.Response(200, json={"studies": None})
    mock_http(monkeypatch, handler)
    client = clinicaltrials.ClinicalTrialsClient()
    trials, source = asyncio.run(client.search(TrialSearchRequest(query="cancer", phase="PHASE2")))
    assert source == "sample-data"
    assert all("PHASE2" in trial.phases and trial.status == "RECRUITING" for trial in trials)
    assert "using sample-data" in caplog.text


@pytest.mark.parametrize("failure", ["constructor", "provider", "empty"])
def test_assistant_fallback(monkeypatch, caplog, failure):
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        enable_llm=True, openai_api_key="test", openai_model="test"))
    class Provider:
        def __init__(self, **kwargs):
            if failure == "constructor":
                raise RuntimeError("test")
            self.responses = SimpleNamespace(create=AsyncMock(
                side_effect=RuntimeError("test") if failure == "provider" else None,
                return_value=SimpleNamespace(output_text="")))
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr(llm, "AsyncOpenAI", Provider)
    request = AssistantRequest(question="Tell me about this trial", trial=Trial(nct_id="NCT1", title="Cancer"))
    response = asyncio.run(llm.TrialAssistant().answer(request))
    assert response.provider == "fallback"
    assert response.answer == llm.TrialAssistant()._fallback_answer(request)
    assert response.sources == ["NCT1"]
    assert "provider_failure" in caplog.text


def test_recommendation_endpoint_contract(monkeypatch):
    from backend.app.api.routes import recommendations
    trial = Trial(nct_id="NCT1", title="Cancer", minimum_age="60 Years", status="RECRUITING")
    monkeypatch.setattr(recommendations.client, "search", AsyncMock(return_value=([trial], "sample-data")))
    with TestClient(app) as client:
        response = client.post("/api/recommendations", json={"condition": "cancer", "age": 20})
        assert client.get("/api/health").status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "sample-data"
    result = body["results"][0]
    assert result["structured_eligibility"]["status"] == "incompatible"
    assert result["structured_eligibility"]["criteria"]["sex"]["state"] == "unknown"
    assert result["explanation"]["manual_review_signals"]
    assert 0 <= result["score"] <= 1


def test_detail_fallback_source_and_frontend_mount(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("offline")
    mock_http(monkeypatch, handler)
    sample = clinicaltrials.ClinicalTrialsClient().sample_trials()[0]
    with TestClient(app) as client:
        response = client.get(f"/api/trials/{sample.nct_id}")
        assert response.status_code == 200
        assert response.json()["source"] == "sample-data"
        assert client.get("/api/trials/nonexistent").status_code == 404
        assert client.get("/").status_code == 200


def test_assistant_success(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        enable_llm=True, openai_api_key="test", openai_model="test"))
    provider = AsyncMock()
    provider.__aenter__.return_value = provider
    provider.responses.create.return_value = SimpleNamespace(output_text="Study details.")
    monkeypatch.setattr(llm, "AsyncOpenAI", lambda **kwargs: provider)
    response = asyncio.run(llm.TrialAssistant().answer(AssistantRequest(
        question="What is this?", trial=Trial(nct_id="1", title="Cancer"))))
    assert response.provider == "openai"
    assert response.answer == "Study details."
