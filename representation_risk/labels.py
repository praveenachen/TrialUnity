"""Label derivation: representation gap and risk level from observed vs. benchmark.

    gap = 1 - Jensen-Shannon similarity(observed, benchmark), in [0, 1]
      0.0 -> observed distribution matches the benchmark exactly
      1.0 -> maximally divergent

risk_level thresholds on `gap`:

    gap <  LOW_MODERATE_THRESHOLD   -> "low"
    gap <  MODERATE_HIGH_THRESHOLD  -> "moderate"
    otherwise                       -> "high"

JS divergence between two distributions that share a common dominant category
(as most real demographic breakdowns do) saturates well below its theoretical
[0, 1] ceiling -- with this project's 5-category fixture benchmark, even a
maximally skewed comparison rarely exceeds ~0.4. 0.02 / 0.06 are round numbers
chosen to land near the 33rd/66th percentile of this project's synthetic fixture
gap distribution (see representation_risk/synthetic_fixture.py), giving a
roughly even three-way split for demo purposes -- they are NOT derived from, or
validated against, real-world representation data, and would need to be
recalibrated against real observed distributions before any production use.

This mirrors, but intentionally does not import, the Jensen-Shannon similarity
formula `backend.app.services.equity.race` uses for ESR's race component. The two
are kept independent on purpose so this experimental ML module never becomes a
hidden dependency of the deterministic ESR path.
"""
import math

LOW_MODERATE_THRESHOLD = 0.02
MODERATE_HIGH_THRESHOLD = 0.06

RISK_LEVELS = ("low", "moderate", "high")


def _js_similarity(observed: dict[str, float], reference: dict[str, float]) -> float:
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
    return max(0.0, min(1.0, 1.0 - js_divergence))


def representation_gap(observed: dict[str, float], benchmark: dict[str, float]) -> float:
    return round(1.0 - _js_similarity(observed, benchmark), 6)


def risk_level_from_gap(gap: float) -> str:
    if gap < LOW_MODERATE_THRESHOLD:
        return "low"
    if gap < MODERATE_HIGH_THRESHOLD:
        return "moderate"
    return "high"
