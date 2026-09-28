"""Deterministic SYNTHETIC historical-trial generator, for pipeline verification only.

Generates completed, result-posted trials with a sampled design/access feature
profile plus a *simulated* observed race/ethnicity enrollment distribution. This
is not real-world data. The simulator encodes an assumed (not evidence-based)
relationship between access-related design features and representation gap --
purely so the training pipeline has learnable signal to exercise end to end.

TrialUnity does not claim this reflects how real trials actually enroll. See
representation_risk/README.md.
"""
import random

from backend.app.models.schemas import Trial
from representation_risk.benchmark_fixture import FIXTURE_BENCHMARK

SEED = 20240601  # fixed so generation is reproducible run to run
NUM_RECORDS = 60

_PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
_CONDITIONS = [
    "Lung Cancer", "Breast Cancer", "Type 2 Diabetes", "Heart Failure",
    "Rheumatoid Arthritis", "Major Depressive Disorder",
]
_REGIONS = [
    "Boston, Massachusetts, United States", "Chicago, Illinois, United States",
    "Toronto, Ontario, Canada", "Seattle, Washington, United States",
    "Miami, Florida, United States", "Denver, Colorado, United States",
    "Remote", "Decentralized - Nationwide",
]
# A deliberately skewed distribution used only as the "worst case" endpoint the
# simulator blends toward -- not a claim about any real trial or population.
# Calibrated (via representation_risk/labels.py's JS-based gap) to actually reach
# the "high" risk band at full blend; a skew that only reshuffles minority shares
# around an already-dominant benchmark category stays too close to the benchmark
# in JS-divergence terms to ever cross the "moderate"/"high" thresholds.
_SKEWED_DISTRIBUTION = {"WHITE": 15.0, "BLACK": 5.0, "ASIAN": 5.0, "HISPANIC_OR_LATINO": 5.0, "OTHER": 70.0}


def _sample_design(rng: random.Random) -> dict:
    num_sites = rng.randint(1, 12)
    locations = (
        rng.sample(_REGIONS, k=num_sites) if num_sites <= len(_REGIONS)
        else [rng.choice(_REGIONS) for _ in range(num_sites)]
    )
    return {
        "phase": rng.choice(_PHASES),
        "locations": locations,
        "target_enrollment": rng.choice([None, rng.randint(20, 600)]),
        "min_age": rng.choice([None, "18 Years", "21 Years", "65 Years"]),
        "max_age": rng.choice([None, "N/A", "75 Years", "85 Years"]),
        "sex": rng.choice(["ALL", "ALL", "ALL", "MALE", "FEMALE", None]),
        "interventions": [f"Intervention {i}" for i in range(rng.randint(1, 3))],
        "eligibility_criteria": (
            "\n".join(f"- Criterion {i}" for i in range(rng.randint(0, 25))) or None
        ),
    }


def _simulated_true_gap(design: dict, rng: random.Random) -> float:
    """Assumed (NOT evidence-based) latent relationship, used only to give the
    synthetic labels learnable signal: fewer sites/regions and no decentralized
    access push the simulated gap up; more sites/regions and decentralized access
    pull it down. This is an invented rule for pipeline verification, not a
    finding about real trials.
    """
    num_sites = len(design["locations"])
    num_regions = len({location.split(",")[-1].strip() if "," in location else location for location in design["locations"]})
    decentralized = any(keyword in location.casefold() for location in design["locations"] for keyword in ("remote", "decentralized"))
    base = 0.55 - 0.03 * num_sites - 0.02 * num_regions - (0.15 if decentralized else 0.0)
    return max(0.0, min(1.0, base + rng.gauss(0, 0.12)))


def _synthesize_observed_distribution(true_gap: float, rng: random.Random) -> dict[str, float]:
    benchmark = FIXTURE_BENCHMARK.distribution
    # Sorted, not a bare set: iteration order here feeds sequential rng.uniform()
    # draws below, and Python's per-process string-hash randomization would
    # otherwise silently reassign which draw lands on which category -- breaking
    # reproducibility across process runs despite the fixed seed.
    categories = sorted(set(benchmark) | set(_SKEWED_DISTRIBUTION))
    blended = {
        category: (1 - true_gap) * benchmark.get(category, 0.0) + true_gap * _SKEWED_DISTRIBUTION.get(category, 0.0)
        for category in categories
    }
    noisy = {category: max(0.1, value * rng.uniform(0.9, 1.1)) for category, value in blended.items()}
    total = sum(noisy.values())
    return {category: round(value / total * 100, 3) for category, value in noisy.items()}


def generate_historical_trials(seed: int = SEED, count: int = NUM_RECORDS) -> list[tuple[Trial, dict[str, float]]]:
    """Returns (trial, observed_race_distribution) pairs. Deterministic given `seed`."""
    rng = random.Random(seed)
    records = []
    for index in range(count):
        design = _sample_design(rng)
        true_gap = _simulated_true_gap(design, rng)
        observed = _synthesize_observed_distribution(true_gap, rng)
        trial = Trial(
            nct_id=f"HIST-{index:04d}",
            title=f"Synthetic Completed Trial {index:04d}",
            status="COMPLETED",
            conditions=[rng.choice(_CONDITIONS)],
            interventions=design["interventions"],
            phases=[design["phase"]],
            locations=design["locations"],
            sex=design["sex"],
            minimum_age=design["min_age"],
            maximum_age=design["max_age"],
            eligibility_criteria=design["eligibility_criteria"],
            target_enrollment=design["target_enrollment"],
            enrollment_race_distribution=observed,
            enrollment_race_source="representation_risk synthetic fixture (SYNTHETIC -- not real data)",
        )
        records.append((trial, observed))
    return records
