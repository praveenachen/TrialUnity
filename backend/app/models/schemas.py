from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.services.equity.schemas import ESRResult
from backend.app.services.representation_risk.schemas import RepresentationRiskPrediction


class PatientProfile(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = None
    condition: str = Field(..., min_length=2)
    location: str | None = None
    intervention_preferences: list[str] = Field(default_factory=list)
    phase_preferences: list[str] = Field(default_factory=list)
    notes: str | None = None


class TrialSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    condition: str | None = None
    location: str | None = None
    phase: str | None = None
    recruitment_status: str | None = "RECRUITING"
    page_size: int = Field(default=10, ge=1, le=50)

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, value: str | None) -> str | None:
        if value is None:
            return None
        phase = value.strip().upper().replace(" ", "").replace("_", "")
        phase = {"EARLYPHASE1": "EARLY_PHASE1", "NOTAPPLICABLE": "NA"}.get(phase, phase)
        if phase not in {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"}:
            raise ValueError("phase must be EARLY_PHASE1, PHASE1–PHASE4, or NA")
        return phase


class TrialSite(BaseModel):
    facility: str | None = None
    location: str
    status: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class Trial(BaseModel):
    nct_id: str
    title: str
    status: str = "Unknown"
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    brief_summary: str | None = None
    eligibility_criteria: str | None = None
    sex: str | None = None
    minimum_age: str | None = None
    maximum_age: str | None = None
    locations: list[str] = Field(default_factory=list)
    trial_sites: list[TrialSite] = Field(default_factory=list)
    sponsor: str | None = None
    source_url: str | None = None
    # Reported participant distributions, represented as non-negative category weights.
    # Values may originate as counts or percentages; ESR normalizes them before comparison.
    enrollment_sex_distribution: dict[str, float] | None = None
    enrollment_race_distribution: dict[str, float] | None = None
    enrollment_sex_source: str | None = None
    enrollment_race_source: str | None = None
    # Planned enrollment size (designModule.enrollmentInfo.count), a pre-outcome design
    # field. Not populated by current ingestion (see representation_risk/README.md);
    # defaults to None so feature extraction has to treat it as legitimately unknown.
    target_enrollment: int | None = None


class MatchExplanation(BaseModel):
    matched_terms: list[str] = Field(default_factory=list)
    eligibility_notes: list[str] = Field(default_factory=list)
    ranking_rationale: str
    patient_friendly_summary: str
    relevant_signals: list[str] = Field(default_factory=list)
    manual_review_signals: list[str] = Field(default_factory=list)


EligibilityState = Literal["compatible", "incompatible", "unknown"]


class EligibilityCriterion(BaseModel):
    state: EligibilityState
    reason: str


class StructuredEligibility(BaseModel):
    status: EligibilityState
    criteria: dict[str, EligibilityCriterion]


class RelevanceScores(BaseModel):
    """Normalized ranking signals; none represent eligibility probability."""

    overall: float = Field(ge=0, le=1, description="Relative ranking score.")
    lexical: float = Field(ge=0, le=1, description="Normalized BM25 relevance.")
    semantic: float = Field(ge=0, le=1, description="Non-negative cosine relevance.")
    structured: dict[str, float] = Field(
        default_factory=dict,
        description="Structured relevance matches on a 0-to-1 scale.",
    )
    weighted_components: dict[str, float] = Field(
        default_factory=dict,
        description="Weighted contributions that sum to overall.",
    )
    weights: dict[str, float] = Field(default_factory=dict)


class TrialRecommendation(BaseModel):
    trial: Trial
    score: float = Field(ge=0, le=1, description="Relative relevance score, not medical eligibility or probability.")
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Weighted contribution of each ranking signal (lexical, semantic, condition, "
        "intervention, phase, location); values sum to `score`.",
    )
    relevance: RelevanceScores
    explanation: MatchExplanation
    structured_eligibility: StructuredEligibility | None = None
    esr: ESRResult | None = Field(
        default=None,
        description="Equity/Access Representation score. Independent of `score` and "
        "`structured_eligibility`; never influences ranking or eligibility.",
    )
    representation_risk: RepresentationRiskPrediction | None = Field(
        default=None,
        description="EXPERIMENTAL predicted under-representation risk, returned only when observed "
        "ESR demographic evidence is unavailable. Never a substitute for `esr`; never influences "
        "ranking, eligibility, or `esr` itself.",
    )


class MatchingFunnel(BaseModel):
    """Truthful stage counts from one search, for the client's retrieval-funnel UI.

    Every count comes directly from candidates actually fetched/evaluated for this
    request -- nothing here is estimated or fabricated. `structured_eligible_trials`
    counts trials whose structured eligibility is not `incompatible` (i.e.
    `compatible` or `unknown`), matching the "needs review" framing used elsewhere;
    it is evaluated over the full candidate set before truncating to `ranked_matches`.
    """

    candidate_trials: int = Field(ge=0, description="Trials fetched for this search, before any filtering.")
    recruiting_trials: int = Field(ge=0, description="Of the candidates, how many have status RECRUITING.")
    structured_eligible_trials: int = Field(
        ge=0, description="Of the candidates, how many are not structurally incompatible."
    )
    ranked_matches: int = Field(ge=0, description="How many results were actually returned (post-limit).")


class TrialSearchResponse(BaseModel):
    query: str
    total: int
    results: list[TrialRecommendation]
    source: str
    funnel: MatchingFunnel


class TrialDetailResponse(BaseModel):
    trial: Trial
    explanation: MatchExplanation | None = None
    source: str | None = None


class AssistantRequest(BaseModel):
    question: str = Field(..., min_length=3)
    trial: Trial
    patient_profile: PatientProfile | None = None


class AssistantResponse(BaseModel):
    answer: str
    grounded: bool = True
    provider: str = "fallback"
    sources: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    llm_enabled: bool = False
    openai_key_configured: bool = False
