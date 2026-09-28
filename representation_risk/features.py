"""Deterministic trial-level feature extraction for the representation-risk model.

Every feature is derived from *design/access* fields that exist before a trial
reports enrollment outcomes. Nothing here reads observed demographics
(`enrollment_*_distribution`), nothing is patient-level, and nothing infers race,
ethnicity, or any other demographic attribute from geography or names.
"""
import math
import re

from backend.app.models.schemas import Trial

FEATURE_NAMES = [
    "phase_ordinal",
    "target_enrollment_log",
    "target_enrollment_known",
    "num_sites",
    "num_regions",
    "age_range_breadth_years",
    "age_range_known",
    "sex_inclusive",
    "sex_reported",
    "decentralized_access",
    "intervention_count",
    "eligibility_criteria_line_count",
    "eligibility_criteria_known",
]

_PHASE_ORDER = {"EARLY_PHASE1": 0, "PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}
_DECENTRALIZED_KEYWORDS = ("remote", "virtual", "decentralized", "telehealth", "online")


def _phase_ordinal(phases: list[str]) -> float:
    values = [_PHASE_ORDER[phase.strip().upper()] for phase in phases if phase.strip().upper() in _PHASE_ORDER]
    return float(max(values)) if values else -1.0


def _region_signature(location: str) -> str:
    parts = [part.strip() for part in location.split(",") if part.strip()]
    return ", ".join(parts[1:]) if len(parts) > 1 else location.strip()


def _age_years(bound: str | None) -> float | None:
    if not bound:
        return None
    text = bound.strip().upper()
    if text == "N/A":
        return None
    match = re.fullmatch(r"(\d+)\s+YEARS?", text)
    return float(match.group(1)) if match else None


def extract_features(trial: Trial) -> dict[str, float]:
    """Pure function of design/access fields only -- same trial in, same features out."""
    locations = [location for location in trial.locations if location and location.strip()]

    target_enrollment = trial.target_enrollment
    target_enrollment_known = target_enrollment is not None and target_enrollment > 0

    minimum_age = _age_years(trial.minimum_age)
    maximum_age = _age_years(trial.maximum_age)
    age_range_known = minimum_age is not None or maximum_age is not None
    age_breadth = (maximum_age if maximum_age is not None else 120.0) - (
        minimum_age if minimum_age is not None else 0.0
    ) if age_range_known else 0.0

    trial_sex = (trial.sex or "").strip().upper()
    sex_reported = trial_sex in {"ALL", "MALE", "FEMALE"}
    sex_inclusive = 1.0 if trial_sex == "ALL" else 0.0

    decentralized = any(
        keyword in location.casefold() for location in locations for keyword in _DECENTRALIZED_KEYWORDS
    )

    criteria_lines = [line for line in (trial.eligibility_criteria or "").splitlines() if line.strip()]

    return {
        "phase_ordinal": _phase_ordinal(trial.phases),
        "target_enrollment_log": math.log1p(target_enrollment) if target_enrollment_known else 0.0,
        "target_enrollment_known": 1.0 if target_enrollment_known else 0.0,
        "num_sites": float(len(locations)),
        "num_regions": float(len({_region_signature(location).casefold() for location in locations})),
        "age_range_breadth_years": float(age_breadth),
        "age_range_known": 1.0 if age_range_known else 0.0,
        "sex_inclusive": sex_inclusive,
        "sex_reported": 1.0 if sex_reported else 0.0,
        "decentralized_access": 1.0 if decentralized else 0.0,
        "intervention_count": float(len(trial.interventions)),
        "eligibility_criteria_line_count": float(len(criteria_lines)),
        "eligibility_criteria_known": 1.0 if trial.eligibility_criteria else 0.0,
    }
