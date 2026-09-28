"""Overall ESR evidence coverage: how much of the full picture is actually backed by evidence."""
from backend.app.services.equity.schemas import ComponentEvidence


def overall_coverage(components: dict[str, ComponentEvidence], weights: dict[str, float]) -> float:
    """Weighted average of each component's own evidence coverage.

    Uses the *full* ESR weights (not renormalized to available components) so that a
    missing high-weight component visibly reduces overall coverage rather than being
    hidden by renormalization. E.g. race (weight 0.40) unavailable but socioeconomic
    and sex fully covered -> coverage = 0.35*1 + 0.25*1 + 0.40*0 = 0.60, not 1.0.
    """
    return round(sum(weights[name] * components[name].evidence_coverage for name in weights), 4)
