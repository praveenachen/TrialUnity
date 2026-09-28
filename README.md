# TrialUnity

TrialUnity is an AI-powered clinical trial matching platform that helps patients and care teams discover, compare, and understand relevant clinical trials. The project has been refactored from a Streamlit prototype into a modern health-tech workflow application with a FastAPI backend, ClinicalTrials.gov ingestion, semantic retrieval, explainable recommendations, and a polished patient-facing frontend.

The goal is not to replace clinicians or study coordinators. TrialUnity is a navigation and decision-support layer that makes trial discovery more transparent, patient-friendly, and easier to reason about.

## Product Overview

TrialUnity supports a practical healthcare AI workflow:

1. A user submits a patient/profile intake form.
2. The backend queries and normalizes ClinicalTrials.gov study records.
3. A hybrid retrieval pipeline ranks trials using lexical (BM25) and semantic (embedding) relevance plus structured matching signals.
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
    retrieval/         Hybrid lexical (BM25) + dense (embedding) relevance ranking
    eligibility.py     Deterministic structured eligibility checks (age/sex/recruitment)
    equity/            Deterministic ESR (equity/access representation) scoring
    recommendations.py Explainable recommendation generation
    llm.py             Grounded optional LLM assistant
    text.py            Text cleanup and token utilities
  data/sample_trials.json Local fallback records for offline development

tests/                Focused backend tests
```

## Retrieval And Recommendation Methodology

The retrieval pipeline is a hybrid of lexical and dense semantic search, combined with weighted structured signals:

- patient condition and notes are converted into the lexical/semantic retrieval query; location, phase, and intervention preferences contribute once through explicit structured signals
- trial titles, conditions, interventions, summaries, phases, and locations become searchable documents (free-text eligibility criteria are excluded so their length or wording can never influence ranking)
- **lexical relevance** comes from BM25 keyword matching over that text
- **semantic relevance** comes from cosine similarity between local SentenceTransformer embeddings (`all-MiniLM-L6-v2`, loaded once per process, no external API)
- condition, intervention, phase, and location contribute additional weighted signals; matched terms are explanatory only
- every recommendation returns normalized lexical, semantic, and structured relevance signals plus their weights and weighted contributions; the weighted sum is the overall relevance score, not an eligibility probability
- structured age, sex, and recruitment checks are returned separately and never affect the relevance score; incompatible results rank after other results, and missing or unsupported data remains unknown
- structured compatibility is not full medical eligibility; free-text criteria always require review

This design can be upgraded to a vector database without changing the API contract if the trial catalog outgrows in-memory ranking.

## Equity / Access Representation (ESR)

Every recommendation also returns an **ESR** score: a deterministic, auditable measure of
representation and access evidence, entirely separate from the relevance score above and
from structured eligibility. ESR never affects ranking or eligibility, and nothing about
it uses machine learning — every number is a plain, inspectable formula over evidence that
is actually present, or `null` when it isn't.

ESR combines three weighted components:

| Component | Weight | What it measures |
| --- | --- | --- |
| Socioeconomic Access | 0.35 | Site count, geographic spread, decentralized/remote participation, and patient-to-site proximity — structural access signals only, never income, transportation, or ability to pay |
| Sex Representation | 0.25 | Protocol sex eligibility (**prospective**) for recruiting trials, or reported enrollment balance vs. a baseline (**observed**) when the registry has published results |
| Race/Ethnicity Representation | 0.40 | Reported enrollment distribution vs. an explicitly supplied benchmark, compared with Jensen-Shannon similarity — never generated without both a real observed distribution and a real benchmark |

Key rules, enforced in code, not just by convention:

- **Missing evidence is never 0 or 100.** A component with insufficient evidence returns `score: null`, an `evidence_type` explaining why (e.g. `insufficient_data`, `insufficient_benchmark`), and the specific `missing_evidence`.
- **The overall score only uses available components**, with weights renormalized over what's available (e.g. race missing → the score is the weighted average of socioeconomic and sex alone).
- **`evidence_coverage` reports completeness separately from the score**, using the *full* weights (a missing high-weight component visibly lowers coverage rather than being hidden by renormalization).
- **`mode`** is `observed` (real reported data), `prospective` (protocol-only, no enrollment results yet), `mixed` (some of each), or `insufficient_data`.
- **Protocol sex inclusivity is explicitly labeled as not observed representation.** A trial open to "ALL" sexes is not the same claim as "enrollment turned out balanced."
- **Race is never inferred from geography, names, or any other proxy.** Without both a reported enrollment distribution and a real, sourced benchmark, the component returns `insufficient_data`/`insufficient_benchmark` rather than a number.

ESR lives in `backend/app/services/equity/` and is exposed as `esr` on each `TrialRecommendation`. Reported ClinicalTrials.gov baseline sex and race/ethnicity distributions are normalized during ingestion. Race scoring additionally requires an explicitly sourced, population-scoped benchmark supplied through the benchmark-provider contract; the production registry is empty by default. A future ranking layer may use ESR as a secondary objective; ESR currently remains separate from relevance ranking.

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

## Retrieval Evaluation

`evaluation/` holds a small, reproducible benchmark proving whether hybrid retrieval actually outranks BM25-only and embeddings-only baselines, with honest results (including where hybrid doesn't win). Run it with `python -m evaluation.run`; see [evaluation/README.md](evaluation/README.md) for the benchmark, metrics, current results, and limitations.

## Representation Risk (Experimental)

`representation_risk/` is an **experimental, not validated** ML layer that predicts an under-representation risk band for recruiting trials that don't yet have observed enrollment demographics -- trained and evaluated entirely on synthetic fixture data, never mixed into relevance, eligibility, or ESR. Run it with `python -m representation_risk.run`; see [representation_risk/README.md](representation_risk/README.md) for the target definition, features, models, results, and limitations.

## Local Setup

### Run With Python

```powershell
# Requires Python 3.10 or newer
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

### Run With Docker

Docker runs the same FastAPI app in one container. The frontend and backend are served together.

```powershell
Copy-Item .env.example .env
docker build -t trialunity .
docker run --env-file .env -p 8001:8001 trialunity
```

Open:

```text
http://127.0.0.1:8001
```

API docs:

```text
http://127.0.0.1:8001/docs
```

To enable OpenAI-backed assistant responses in Docker, set these in `.env` before running the container:

```env
ENABLE_LLM=true
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Do not commit `.env`; it is ignored by Git and Docker.

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

- Hybrid BM25 + local SentenceTransformer retrieval was chosen over a single method: BM25 covers exact keyword/acronym matches embeddings can miss, embeddings cover paraphrase and synonym matches keyword search misses. The embedding model runs locally and loads once per process, so there is no external API dependency or per-request reload cost.
- Free-text eligibility criteria are deliberately excluded from the search index, keeping ranking relevance and eligibility judgment independent as a matter of architecture, not just convention.
- The ClinicalTrials.gov client falls back to local sample data so demos do not fail when offline.
- The frontend is static HTML/CSS/JS served by FastAPI to avoid unnecessary build tooling.
- The old Streamlit prototype entrypoint is retained only as a migration note.
- Recommendation scores are transparent weighted signals, not opaque medical eligibility decisions.
- ESR is deterministic and formula-based on purpose, not ML-scored: representation/access claims need to be auditable, and "why did this trial get this equity score" must always be answerable from a rationale string and a named source, not a model weight.
- ESR's Socioeconomic Access component scores site/geographic structure only, because that's the only access-relevant data TrialUnity's sources actually contain; it does not claim to model a patient's real socioeconomic status.
- ESR's Race/Ethnicity component refuses to score without both a real reported enrollment distribution and an explicitly supplied, sourced benchmark, rather than fabricating or hardcoding a population reference.

## Future Improvements

- Add a small vector index if the trial catalog grows past in-memory-ranking scale.
- Cache ClinicalTrials.gov query results for faster repeat searches.
- Explore additional validated structured eligibility checks.
- Add trial comparison workflows.
- Add CI.
- Add clinician/researcher views for cohort diversity and recruitment planning.
- **Remaining ESR data gaps:** source and configure defensible condition/location-specific race and ethnicity reference distributions through `RaceBenchmarkProvider`; quantify how frequently posted results contain usable baseline demographics; reconcile studies that publish race and ethnicity as separate dimensions; and consider an external area-deprivation or travel-time dataset to extend Socioeconomic Access beyond site/geography signals.
