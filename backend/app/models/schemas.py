from pydantic import BaseModel, Field


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


class TrialRecommendation(BaseModel):
    trial: Trial
    score: float = Field(ge=0, le=1)
    explanation: MatchExplanation


class TrialSearchResponse(BaseModel):
    query: str
    total: int
    results: list[TrialRecommendation]
    source: str


class TrialDetailResponse(BaseModel):
    trial: Trial
    explanation: MatchExplanation | None = None


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
