"""Response contract for Equity/Access Representation (ESR) scoring.

ESR is deliberately separate from clinical relevance (backend.app.services.retrieval)
and from structured eligibility (backend.app.services.eligibility). It never feeds
into either: a trial's ESR score must not change its rank, and a trial's relevance
or eligibility must not change its ESR score.
"""
from typing import Literal

from pydantic import BaseModel, Field

EvidenceMode = Literal["observed", "prospective", "mixed", "insufficient_data"]

# What kind of evidence produced a component's score, in plain terms:
#   observed_enrollment               - real reported participant counts
#   observed_geographic                - real site/location data (not speculative)
#   observed_distribution_comparison   - real enrollment distribution vs. a real benchmark
#   protocol_inclusivity               - protocol eligibility text only (prospective, not observed)
#   insufficient_data                  - no usable evidence at all
#   insufficient_benchmark             - evidence exists but no valid reference to compare it to
EvidenceType = Literal[
    "observed_enrollment",
    "observed_geographic",
    "observed_distribution_comparison",
    "protocol_inclusivity",
    "insufficient_data",
    "insufficient_benchmark",
]


class ComponentEvidence(BaseModel):
    """One ESR component's score plus the evidence trail behind it.

    `score` is None whenever there isn't enough evidence to responsibly produce a
    number -- it is never coerced to 0 or 100 for missing data.
    """

    score: float | None = Field(default=None, ge=0, le=100)
    evidence_coverage: float = Field(ge=0, le=1, description="How complete the evidence behind this score is.")
    evidence_type: EvidenceType
    rationale: str
    source: str | None = None
    missing_evidence: list[str] = Field(default_factory=list)


class ESRResult(BaseModel):
    """Equity/access Score for one (patient, trial) pair. Deterministic, no ML."""

    score: float | None = Field(default=None, ge=0, le=100)
    mode: EvidenceMode
    evidence_coverage: float = Field(ge=0, le=1)
    components: dict[str, ComponentEvidence]
    weights_used: dict[str, float]
