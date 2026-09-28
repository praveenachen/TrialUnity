"""Prediction contract for the EXPERIMENTAL Representation Risk model.

This is a prediction, not observed evidence, and never a substitute for ESR
(backend.app.services.equity). `evidence_type` is always "predicted" so nothing
downstream can mistake it for deterministic, evidence-based ESR.
"""
from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "moderate", "high"]


class RepresentationRiskPrediction(BaseModel):
    risk_level: RiskLevel | None = Field(
        default=None, description="Predicted under-representation risk band, or null if unavailable."
    )
    probabilities: dict[str, float] | None = Field(
        default=None, description="Per-class predicted probabilities, when the model supports them."
    )
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_version: str
    evidence_type: Literal["predicted"] = "predicted"
    drivers: list[str] = Field(default_factory=list, description="Top contributing features, most important first.")
    limitations: list[str] = Field(default_factory=list)
