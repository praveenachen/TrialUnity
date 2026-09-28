"""Sex Representation: protocol inclusivity (prospective) vs. reported enrollment (observed).

These are never conflated: a trial open to "ALL" sexes at the protocol level says
nothing about how enrollment actually turned out, and the response always labels
which one it is returning.
"""
import math

from backend.app.models.schemas import Trial
from backend.app.services.equity.schemas import ComponentEvidence

# Generic 50/50 baseline used when no disease-specific benchmark is supplied. Swappable
# per call via `target_ratio` without touching the scoring function itself.
DEFAULT_TARGET_RATIO = {"MALE": 0.5, "FEMALE": 0.5}


def _total_variation_score(observed: dict[str, float], target: dict[str, float]) -> float:
    """1 - total variation distance between two category distributions, on a 0-100 scale.

    Identical distributions -> 100. Completely disjoint distributions -> 0. Isolated on
    purpose so a disease-specific benchmark/metric can replace this function later without
    touching the prospective path or the ESR service.
    """
    categories = set(observed) | set(target)
    total_variation = 0.5 * sum(abs(observed.get(category, 0.0) - target.get(category, 0.0)) for category in categories)
    return round(max(0.0, min(1.0, 1.0 - total_variation)) * 100, 2)


def prospective_sex_component(trial: Trial) -> ComponentEvidence:
    """Protocol sex eligibility only. Always available when the registry reports it."""
    trial_sex = (trial.sex or "").strip().upper()
    if trial_sex not in {"ALL", "MALE", "FEMALE"}:
        return ComponentEvidence(
            score=None,
            evidence_coverage=0.0,
            evidence_type="insufficient_data",
            rationale="Trial does not specify a protocol sex eligibility value.",
            source="ClinicalTrials.gov eligibilityModule.sex",
            missing_evidence=["Protocol sex eligibility not reported."],
        )

    if trial_sex == "ALL":
        score = 100.0
        detail = "Protocol eligibility is open to all sexes."
    else:
        score = 0.0
        detail = (
            f"Protocol eligibility is restricted to {trial_sex.title()} participants only. This may be "
            "clinically appropriate (e.g. sex-specific conditions) -- this score is not a quality judgment."
        )
    return ComponentEvidence(
        score=score,
        evidence_coverage=1.0,
        evidence_type="protocol_inclusivity",
        rationale=f"PROSPECTIVE (protocol inclusivity, not observed participant balance): {detail}",
        source="ClinicalTrials.gov eligibilityModule.sex",
        missing_evidence=[],
    )


def observed_sex_component(trial: Trial, target_ratio: dict[str, float] | None = None) -> ComponentEvidence | None:
    """Reported enrollment balance, if the registry published results. None if not available."""
    distribution = trial.enrollment_sex_distribution
    if not distribution or not all(
        math.isfinite(value) and value >= 0 for value in distribution.values()
    ) or sum(distribution.values()) <= 0:
        return None

    target = target_ratio or DEFAULT_TARGET_RATIO
    total = sum(distribution.values())
    observed_fractions = {category.upper(): count / total for category, count in distribution.items()}
    score = _total_variation_score(observed_fractions, target)
    return ComponentEvidence(
        score=score,
        evidence_coverage=1.0,
        evidence_type="observed_enrollment",
        rationale=(
            "OBSERVED reported participant sex/gender enrollment compared against a baseline target "
            f"of {target}. This baseline is generic and swappable; a disease-specific benchmark can "
            "replace it without changing this scoring path."
        ),
        source=trial.enrollment_sex_source or "Reported participant enrollment",
        missing_evidence=[],
    )
