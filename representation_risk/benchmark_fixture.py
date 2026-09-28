"""A disclosed SYNTHETIC reference distribution, for pipeline verification only.

This is not a real epidemiological, census, or disease-prevalence benchmark --
TrialUnity has no source for one and does not fabricate one. It exists solely so
this experiment has something concrete to compare synthetic historical
enrollment against, exercising the same label-derivation formula a real benchmark
would go through. Swapping in a real, sourced benchmark (comparable to
`backend.app.services.equity.benchmarks.RaceBenchmark`, the production ESR path)
is a data task for a later phase, not something this experiment does.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    distribution: dict[str, float]
    label: str


FIXTURE_BENCHMARK = Benchmark(
    distribution={"WHITE": 60.0, "BLACK": 13.0, "ASIAN": 6.0, "HISPANIC_OR_LATINO": 18.0, "OTHER": 3.0},
    label="representation_risk fixture benchmark v1 (SYNTHETIC -- not a real population reference)",
)
