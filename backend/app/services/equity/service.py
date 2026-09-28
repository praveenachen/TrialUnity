"""ESR = Socioeconomic Access + Sex Representation + Race/Ethnicity Representation.

Deterministic and auditable by design: no ML, no LLM. Every score is either a plain
formula over available evidence or `None` when there isn't enough evidence -- never a
default of 0 or 100 standing in for "unknown". ESR is computed independently of, and
never mixed into, clinical relevance (backend.app.services.retrieval) or structured
eligibility (backend.app.services.eligibility).
"""
import math

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.equity.coverage import overall_coverage
from backend.app.services.equity.race import RaceBenchmark, observed_race_component
from backend.app.services.equity.schemas import ComponentEvidence, ESRResult
from backend.app.services.equity.sex import observed_sex_component, prospective_sex_component
from backend.app.services.equity.socioeconomic import socioeconomic_component

# One obvious place to tune ESR's weighting. Sums to 1.0, enforced at import time.
WEIGHTS = {
    "socioeconomic": 0.35,
    "sex": 0.25,
    "race": 0.40,
}
if not math.isclose(sum(WEIGHTS.values()), 1.0):
    raise RuntimeError("ESR weights must sum to 1.0")

# Which overall `mode` an available component's evidence_type counts as. Component types
# not listed here (insufficient_data, insufficient_benchmark) contribute no score and no mode.
_MODE_BY_EVIDENCE_TYPE = {
    "observed_enrollment": "observed",
    "observed_geographic": "observed",
    "observed_distribution_comparison": "observed",
    "protocol_inclusivity": "prospective",
}


def compute_esr(
    patient: PatientProfile,
    trial: Trial,
    *,
    race_benchmark: RaceBenchmark | None = None,
    sex_target_ratio: dict[str, float] | None = None,
) -> ESRResult:
    components: dict[str, ComponentEvidence] = {
        "socioeconomic": socioeconomic_component(patient, trial),
        "sex": observed_sex_component(trial, sex_target_ratio) or prospective_sex_component(trial),
        "race": observed_race_component(trial, race_benchmark),
    }

    available = {name: component for name, component in components.items() if component.score is not None}
    available_weight = sum(WEIGHTS[name] for name in available)
    score = None
    if available_weight > 0:
        weighted_sum = sum(WEIGHTS[name] * component.score for name, component in available.items())
        # Normalize over only the available weights so missing evidence never drags the
        # score toward 0 -- it only reduces `evidence_coverage` below.
        score = round(weighted_sum / available_weight, 2)

    modes = {
        _MODE_BY_EVIDENCE_TYPE[component.evidence_type]
        for component in components.values()
        if component.evidence_type in _MODE_BY_EVIDENCE_TYPE
    }
    if not modes:
        mode = "insufficient_data"
    elif modes == {"observed"}:
        mode = "observed"
    elif modes == {"prospective"}:
        mode = "prospective"
    else:
        mode = "mixed"

    return ESRResult(
        score=score,
        mode=mode,
        evidence_coverage=overall_coverage(components, WEIGHTS),
        components=components,
        weights_used=WEIGHTS,
    )
