# Representation Risk (EXPERIMENTAL, NOT VALIDATED)

> **This is a prediction, not evidence.** It is trained and evaluated entirely on
> synthetic fixture data, not real historical trials, and must never be treated as
> a validated real-world claim. It is never a substitute for observed ESR (Equity
> / Access Representation, `backend.app.services.equity`) and never influences
> clinical relevance ranking or structured eligibility.

Run the offline training/evaluation experiment with:

```bash
python -m representation_risk.run
```

No network access is used or required — everything is a local, deterministic fixture.

## Why this exists

ESR (`backend.app.services.equity`) can only score race/ethnicity representation
once a trial has *reported* enrollment results, which recruiting trials don't
have yet. This experiment asks: for a **recruiting trial with no observed
demographics**, can trial *design* alone (phase, site count, age range, etc.)
predict whether it's likely to end up under-representing groups relative to a
benchmark, the way completed/reported trials sometimes do? It's a prediction to
flag for attention, not a number to rank trials by.

## Target definition

For each **historical (completed, result-posted) trial**, the label is derived
from its observed demographic distribution compared against a benchmark:

```
gap = 1 - Jensen-Shannon similarity(observed_distribution, benchmark_distribution)   # in [0, 1]

risk_level:
    gap < 0.02  -> "low"
    gap < 0.06  -> "moderate"
    otherwise   -> "high"
```

Same Jensen-Shannon-divergence approach ESR's race component uses (see
`backend.app.services.equity.race`), but **reimplemented independently** in
`representation_risk/labels.py` — deliberately not imported from `equity`, so
this experimental module can never become a hidden dependency of the
deterministic ESR path.

The 0.02 / 0.06 cut points are round numbers chosen to land near the 33rd/66th
percentile of *this project's own synthetic fixture* gap distribution (see
[Why the thresholds look small](#why-the-thresholds-look-small)) — **not derived
from or validated against real-world representation data.**

## Feature list (trial-level design/access only, no leakage)

All 13 features come from `representation_risk/features.py:extract_features`,
computed purely from fields that exist **before** a trial reports outcomes:

| Feature | What it captures |
| --- | --- |
| `phase_ordinal` | Trial phase, ordered EARLY_PHASE1(0) .. PHASE4(4), or -1 if unspecified |
| `target_enrollment_log` / `target_enrollment_known` | log1p(planned enrollment size), with a missingness flag |
| `num_sites` | Number of listed site locations |
| `num_regions` | Number of distinct state/country groupings among those sites |
| `age_range_breadth_years` / `age_range_known` | Max age minus min age (years), with a missingness flag |
| `sex_inclusive` / `sex_reported` | Whether protocol eligibility is open to all sexes, with a "was sex eligibility even reported" flag |
| `decentralized_access` | Whether any site location mentions remote/virtual/decentralized/telehealth/online participation |
| `intervention_count` | Number of listed interventions |
| `eligibility_criteria_line_count` / `eligibility_criteria_known` | Non-empty lines in the free-text eligibility criteria (a rough structural burden proxy), with a missingness flag |

**Deliberately excluded** (per the brief's constraints): patient-level data,
names or geography as demographic proxies, inferred race/ethnicity, observed
enrollment/demographic fields (that would be direct target leakage — see
`tests/test_representation_risk.py::test_feature_extraction_never_reads_observed_demographics`),
opaque text embeddings, and study/intervention *type* (Drug/Device/Behavioral),
which isn't tracked anywhere in TrialUnity's `Trial` schema — a known gap, not
an oversight (see [Limitations](#limitations)).

`target_enrollment` is a new optional `Trial` field. Current ClinicalTrials.gov
ingestion doesn't populate it yet (same "field exists, ingestion doesn't wire it
yet" pattern as `enrollment_sex_distribution` before it was wired) — features
built from it degrade gracefully via the `*_known` flags rather than crashing or
silently imputing a misleading value.

## Data reality: synthetic, not real

**There is no real historical labeled trial dataset in this repo.** Per the
brief's guidance for that situation, `representation_risk/synthetic_fixture.py`
generates 60 clearly-labeled **synthetic** "completed" trials with a fixed random
seed (`SEED = 20240601`, reproducible run to run):

1. Sample design features (phase, site count/locations, age bounds, sex,
   interventions, eligibility criteria line count, target enrollment) from
   simple, documented distributions.
2. Compute a **simulated latent "true gap"** from those features using an
   **invented, not evidence-based** rule (fewer sites/regions and no
   decentralized access → higher simulated gap) plus Gaussian noise — this
   exists purely to give the synthetic labels *some* learnable structure so the
   pipeline has something non-trivial to fit; it is not a claim about how real
   trials actually enroll.
3. Blend a fixture benchmark distribution toward a deliberately skewed
   "worst case" distribution by that simulated gap, add multiplicative noise,
   and treat that as the trial's "observed" distribution.
4. Derive the actual label from step 3's distribution using the **real**
   `representation_gap` / `risk_level_from_gap` formula (not directly from the
   latent value in step 2) — the label-derivation step is exercised exactly as
   it would be on real reported data.

The reference "benchmark" (`representation_risk/benchmark_fixture.py`) is
likewise explicitly disclosed as `SYNTHETIC -- not a real population reference`.
**No demographic data — synthetic or real — was scraped, fabricated as if real,
or presented as a production benchmark.**

### Why the thresholds look small

Jensen-Shannon divergence between two distributions sharing a dominant category
(as most demographic breakdowns do) saturates well below its theoretical ceiling.
With this project's 5-category fixture benchmark, even a maximally skewed
comparison rarely exceeds a gap of ~0.4, and typical synthetic trials land far
below that — hence 0.02/0.06, not the more "textbook-looking" 0.15/0.35 an
uncalibrated reader might expect. This is disclosed here specifically so nobody
mistakes the small numbers for a bug.

## Models and evaluation results

Two lightweight sklearn baselines, no deep learning:

- **Logistic regression** (standardized features, `max_iter=1000`) — coefficients
  (mean absolute value across classes) as the importance proxy.
- **Random forest** (`n_estimators=200, max_depth=5`, fixed `random_state=42`) —
  built-in `feature_importances_`.

70/30 deterministic train/test split (`random_state=42`, stratified by risk
level when class sizes allow it). Metrics reported for classification, per the
brief: **macro F1**, **balanced accuracy**, and the **confusion matrix**.
Regression wasn't used because the 3-band label is the more interview-explainable
target and matches ESR's own ordinal framing (compatible/incompatible/unknown-style
bands), not because a continuous target wouldn't work.

_(Generated by `python -m representation_risk.run`; see
`representation_risk/results/latest.json`/`latest.md` for the full output,
including per-feature importances.)_

| model | macro F1 | balanced accuracy |
| --- | --- | --- |
| logistic_regression | 0.4306 | 0.5333 |
| random_forest | 0.5556 | 0.6000 |

**Neither model is hard-coded as "the winner."** `representation_risk/run.py`
prints both; the backend predictor (`backend/app/services/representation_risk/predictor.py`)
picks whichever scores higher macro F1 on this held-out split — here, random
forest. That's a real (if small-sample) result, not something tuned to make one
model win: on a 45-row training set with only 13 features, neither model
separates "moderate" cleanly from the other two bands (see the confusion
matrices in `representation_risk/results/latest.md`), which is exactly the kind
of honest, unimpressive finding a tiny fixture experiment is supposed to
surface, not hide.

Both models still comfortably clear the "no better than guessing" bar (roughly
0.33 macro F1 for 3 balanced classes) — see
[Prediction contract and safe unavailability](#prediction-contract-and-safe-unavailability).

## Explainability

- **Logistic regression**: mean absolute coefficient per feature (linear, so a
  coefficient's sign and magnitude are directly interpretable).
- **Random forest**: built-in impurity-based `feature_importances_`.

No SHAP — the brief calls for it only if clearly necessary, and for 13
hand-designed features on two small, simple models, plain
coefficients/importances are already fully inspectable. Each prediction also
carries its own top-3 `drivers` (see the prediction contract below).

## Prediction contract and safe unavailability

`backend/app/services/representation_risk/schemas.py`:

```json
{
  "risk_level": "low" | "moderate" | "high" | null,
  "probabilities": {"low": 0.1, "moderate": 0.7, "high": 0.2} | null,
  "confidence": 0.7,
  "model_version": "representation-risk-experimental-v1",
  "evidence_type": "predicted",
  "drivers": ["num_sites (importance 0.55)", "..."],
  "limitations": ["EXPERIMENTAL and NOT VALIDATED: ...", "..."]
}
```

`evidence_type` is always the literal `"predicted"`, so nothing downstream can
mistake this for ESR's deterministic, evidence-based scoring.

If the selected model's fixture-evaluation macro F1 falls below
`MIN_MACRO_F1_FOR_AVAILABILITY` (0.34, roughly chance level for 3 balanced
classes), `predict()` returns `risk_level: null`, `probabilities: null`,
`confidence: null`, and a `limitations` entry explaining why — never a
low-confidence guess dressed up as a real prediction.

## Integration boundary

- `TrialRecommendation.representation_risk` (see `backend/app/models/schemas.py`)
  is a field entirely separate from `score`, `score_breakdown`, `relevance`,
  `structured_eligibility`, and `esr`.
- `backend/app/services/recommendations.py` never uses it for sorting, filtering,
  or scoring — trial order is unaffected either way.
- `predict_representation_risk(trial)` (in
  `backend/app/services/representation_risk/predictor.py`) returns `None`
  whenever `trial.enrollment_race_distribution` is present — i.e. whenever ESR
  already has real, authoritative evidence — so a prediction can never appear
  alongside (let alone override) observed ESR data.
- The predictor takes **no patient argument at all**: it is purely a function of
  the trial's design fields, so it cannot vary with, or be confused for, a
  patient-specific relevance or eligibility judgment.

## Tests

`tests/test_representation_risk.py` covers: no target leakage in feature
extraction, deterministic feature extraction and train/test split, the
prediction schema, safe "unavailable" behavior when model quality is
insufficient, independence from ESR/relevance/eligibility, observed ESR never
being overwritten by a prediction, and the synthetic pipeline running fully
offline.

## Limitations before any production use

- **Not validated on real data.** Every number in this document comes from a
  60-row *synthetic* fixture with an *invented* generating rule — it says
  nothing about real trial enrollment patterns.
- **No real reference benchmark.** `benchmark_fixture.py`'s distribution is
  disclosed as synthetic; a real, sourced benchmark (the same kind ESR's
  `RaceBenchmark` expects) would need to be substituted before labels meant
  anything.
- **Ingestion doesn't populate `target_enrollment` yet** — every real trial the
  app serves today will have `target_enrollment_known = 0` for that feature.
- **`study_type`/`intervention_model`** (Drug vs. Device vs. Behavioral, etc.),
  suggested in the original brief, isn't in `Trial` at all yet, so it's not a
  feature here — a real gap, not a silent omission.
- **60 rows is far too small a sample** for the confusion matrix or F1 numbers
  above to generalize; they demonstrate the pipeline runs correctly, not that
  either model "works."
- **Risk-level thresholds are calibrated to this fixture's own gap
  distribution**, not to any real-world notion of "low"/"moderate"/"high" risk.
- **Race/ethnicity only.** This experiment doesn't attempt sex-representation
  risk prediction (ESR's sex component already has a real prospective/observed
  split that doesn't need one).
