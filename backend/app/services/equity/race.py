"""Race/Ethnicity Representation: observed enrollment vs. an explicitly supplied benchmark.

TrialUnity does not scrape, fabricate, or hardcode population/disease-prevalence race
benchmarks -- there is no source for that in this codebase. A real benchmark must be
supplied by the caller as a `RaceBenchmark`; without one, this always returns
"insufficient_benchmark" rather than inventing a comparison. Participant race/ethnicity
is also never inferred from site geography, patient location, or names.
"""
import math

from backend.app.models.schemas import Trial
from backend.app.services.equity.benchmarks import RaceBenchmark
from backend.app.services.equity.schemas import ComponentEvidence

# Two independent inputs are required for a race score: the observed distribution and a
# benchmark to compare it to. When only the observed side exists, coverage reflects that
# half of the comparison's required evidence is present.
_COVERAGE_WITH_OBSERVED_BUT_NO_BENCHMARK = 0.5


def _valid_distribution(distribution: dict[str, float] | None) -> bool:
    return bool(
        distribution
        and all(math.isfinite(value) and value >= 0 for value in distribution.values())
        and sum(distribution.values()) > 0
    )


def _canonical_distribution(distribution: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for category, value in distribution.items():
        key = category.strip().upper()
        if key:
            result[key] = result.get(key, 0.0) + value
    return result


def _jensen_shannon_similarity(observed: dict[str, float], reference: dict[str, float]) -> float:
    """1 - Jensen-Shannon divergence (base 2, bounded [0, 1]) as a 0-1 similarity score.

    JS divergence is symmetric, always finite (unlike KL divergence), and handles
    categories present in only one distribution without special-casing zeros.
    """
    categories = sorted(set(observed) | set(reference))
    observed_total = sum(max(0.0, observed.get(category, 0.0)) for category in categories)
    reference_total = sum(max(0.0, reference.get(category, 0.0)) for category in categories)
    if observed_total <= 0 or reference_total <= 0:
        return 0.0

    p = [max(0.0, observed.get(category, 0.0)) / observed_total for category in categories]
    q = [max(0.0, reference.get(category, 0.0)) / reference_total for category in categories]
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

    def kl_divergence(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0)

    js_divergence = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return round(max(0.0, min(1.0, 1.0 - js_divergence)), 6)


def observed_race_component(trial: Trial, benchmark: RaceBenchmark | None) -> ComponentEvidence:
    distribution = trial.enrollment_race_distribution
    if not _valid_distribution(distribution):
        return ComponentEvidence(
            score=None,
            evidence_coverage=0.0,
            evidence_type="insufficient_data",
            rationale="No reported participant race/ethnicity enrollment data is available for this trial.",
            source=trial.enrollment_race_source or "Reported participant enrollment",
            missing_evidence=["Participant race/ethnicity enrollment not reported."],
        )

    if not benchmark or not benchmark.is_valid() or not _valid_distribution(benchmark.distribution):
        return ComponentEvidence(
            score=None,
            evidence_coverage=_COVERAGE_WITH_OBSERVED_BUT_NO_BENCHMARK,
            evidence_type="insufficient_benchmark",
            rationale=(
                "Participant race/ethnicity enrollment is reported, but no reference/benchmark "
                "distribution was supplied to compare it against. TrialUnity does not fabricate or "
                "scrape population benchmarks -- a real, sourced benchmark must be supplied explicitly."
            ),
            source=trial.enrollment_race_source or "Reported participant enrollment",
            missing_evidence=["No reference/benchmark race distribution provided."],
        )

    similarity = _jensen_shannon_similarity(
        _canonical_distribution(distribution),
        _canonical_distribution(benchmark.distribution),
    )
    return ComponentEvidence(
        score=round(similarity * 100, 2),
        evidence_coverage=1.0,
        evidence_type="observed_distribution_comparison",
        rationale=(
            f"OBSERVED participant race/ethnicity distribution compared against benchmark "
            f"'{benchmark.provenance()}' using Jensen-Shannon similarity "
            "(1 - normalized JS divergence, base 2)."
        ),
        source=(
            f"{trial.enrollment_race_source or 'Reported participant enrollment'}; "
            f"benchmark: {benchmark.provenance()}"
        ),
        missing_evidence=[],
    )
