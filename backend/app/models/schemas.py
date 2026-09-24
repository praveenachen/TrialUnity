from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
    sponsor: str | None = None
    source_url: str | None = None


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


class TrialRecommendation(BaseModel):
    trial: Trial
    score: float = Field(ge=0, le=1, description="Relative relevance score, not medical eligibility or probability.")
    explanation: MatchExplanation
    structured_eligibility: StructuredEligibility | None = None


class TrialSearchResponse(BaseModel):
    query: str
    total: int
    results: list[TrialRecommendation]
    source: str


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
