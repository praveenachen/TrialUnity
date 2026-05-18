# TrialUnity

TrialUnity is an AI-powered clinical trial matching platform that helps patients and care teams discover, compare, and understand relevant clinical trials. The project has been refactored from a Streamlit prototype into a modern health-tech workflow application with a FastAPI backend, ClinicalTrials.gov ingestion, semantic retrieval, explainable recommendations, and a polished patient-facing frontend.

The goal is not to replace clinicians or study coordinators. TrialUnity is a navigation and decision-support layer that makes trial discovery more transparent, patient-friendly, and easier to reason about.

## Product Overview

TrialUnity supports a practical healthcare AI workflow:

1. A user submits a patient/profile intake form.
2. The backend queries and normalizes ClinicalTrials.gov study records.
3. A retrieval pipeline ranks trials using semantic similarity and structured matching signals.
4. Each result includes a patient-friendly summary, ranking rationale, matched terms, and eligibility considerations.
5. An optional LLM assistant can explain eligibility or trial details while staying grounded in the selected trial record.

## Architecture

```text
frontend/
  index.html          Patient intake, ranked trial cards, details, AI explanation UI
  styles.css          Modern healthcare workflow styling
  app.js              API calls and client-side interaction

backend/app/
  main.py             FastAPI app, CORS, static frontend mounting
  api/routes/         Health, trials, recommendations, assistant endpoints
  core/config.py      Environment-based settings
  models/schemas.py   Pydantic request/response models
  services/
    clinicaltrials.py ClinicalTrials.gov API v2 client and normalization
    retrieval.py      Semantic retrieval and weighted scoring
    recommendations.py Explainable recommendation generation
    llm.py            Grounded optional LLM assistant
    text.py           Text cleanup and token utilities
  data/sample_trials.json Local fallback records for offline development

tests/                Focused backend tests
```

## Retrieval And Recommendation Methodology

The current retrieval pipeline uses a lightweight TF-IDF semantic baseline with cosine similarity. This keeps the project easy to run locally while still demonstrating the same system design used by embedding-based retrieval systems:

- patient profile text is converted into a retrieval query
- trial titles, conditions, interventions, summaries, eligibility criteria, and locations become searchable documents
- semantic relevance is combined with weighted structured signals
- condition, intervention, phase, location, and matched-term overlap contribute to the final score
- every recommendation returns a clear explanation rather than only a numeric rank

This design can be upgraded to sentence-transformer embeddings or a vector database without changing the API contract.

## ClinicalTrials.gov Integration

TrialUnity uses the modern ClinicalTrials.gov API v2 endpoints:

- `GET /api/v2/studies` for study search
- `GET /api/v2/studies/{nctId}` for trial details

The ingestion layer normalizes inconsistent or missing fields into an internal `Trial` schema used by downstream ranking and AI services. If the external API is unavailable during local development, the backend falls back to curated sample records so the app remains usable.

References:

- ClinicalTrials.gov API overview: https://clinicaltrials.gov/data-about-studies/learn-about-api
- API migration guide: https://clinicaltrials.gov/data-api/about-api/api-migration
- Search areas: https://clinicaltrials.gov/data-api/about-api/search-areas

## AI And LLM Layer

The assistant service is optional and environment-controlled. Without an API key, TrialUnity still produces deterministic grounded explanations from trial data. With `ENABLE_LLM=true` and `OPENAI_API_KEY` set, the assistant uses the configured model to answer patient-friendly questions from the selected trial record only.

Trust-oriented constraints:

- answers are grounded in the trial object passed to the service
- missing information should be stated instead of invented
- output is framed as navigation support, not medical advice
- source identifiers and ClinicalTrials.gov links are returned with responses

## API Endpoints

After starting the app, interactive docs are available at `/docs`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service status |
| `POST` | `/api/trials/search` | Search ClinicalTrials.gov and rank results |
| `GET` | `/api/trials/{nct_id}` | Retrieve a normalized trial detail |
| `POST` | `/api/recommendations` | Generate profile-based recommendations |
| `POST` | `/api/assistant/answer` | Explain a trial using grounded AI assistance |

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

The launcher prints the local links:

```text
Frontend: http://127.0.0.1:8001
API docs: http://127.0.0.1:8001/docs
```

TrialUnity uses one FastAPI server for both the frontend and backend. The frontend is served from `frontend/`, and the API lives under `/api`.

If port `8001` is busy, choose another port:

```powershell
$env:PORT=8010
python run.py
```

You can also run Uvicorn directly:

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001
```

Run tests:

```powershell
python -m pytest
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `TrialUnity` | App label |
| `APP_ENV` | `development` | Runtime environment |
| `CTGOV_BASE_URL` | `https://clinicaltrials.gov/api/v2` | ClinicalTrials.gov API base URL |
| `ENABLE_LLM` | `false` | Enables provider-backed assistant responses |
| `OPENAI_API_KEY` | empty | Optional API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Optional assistant model |

## Engineering Tradeoffs

- TF-IDF retrieval was chosen as a reliable local baseline. It is explainable, dependency-light, and easy to replace with embeddings later.
- The ClinicalTrials.gov client falls back to local sample data so demos do not fail when offline.
- The frontend is static HTML/CSS/JS served by FastAPI to avoid unnecessary build tooling.
- The old Streamlit prototype entrypoint is retained only as a migration note.
- Recommendation scores are transparent weighted signals, not opaque medical eligibility decisions.

## Future Improvements

- Add sentence-transformer embeddings and a small vector index.
- Cache ClinicalTrials.gov query results for faster repeat searches.
- Add richer eligibility parsing for age, sex, geography, and biomarkers.
- Add trial comparison workflows.
- Add Docker support and CI.
- Add clinician/researcher views for cohort diversity and recruitment planning.
