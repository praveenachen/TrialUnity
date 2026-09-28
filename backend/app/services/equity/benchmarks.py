"""Contracts for explicitly sourced race/ethnicity reference distributions."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RaceBenchmark:
    distribution: dict[str, float]
    source: str
    population_scope: str = "unspecified"
    condition_scope: str | None = None
    location_scope: str | None = None
    version: str | None = None
    year: int | None = None

    @property
    def source_label(self) -> str:
        return self.source

    def provenance(self) -> str:
        scope = [f"population={self.population_scope}"]
        if self.condition_scope:
            scope.append(f"condition={self.condition_scope}")
        if self.location_scope:
            scope.append(f"location={self.location_scope}")
        if self.version:
            scope.append(f"version={self.version}")
        if self.year:
            scope.append(f"year={self.year}")
        return f"{self.source} ({'; '.join(scope)})"

    def is_valid(self) -> bool:
        return bool(
            self.source.strip()
            and self.population_scope.strip().casefold() not in {"", "unspecified"}
        )


class RaceBenchmarkProvider(Protocol):
    def get_race_benchmark(
        self, *, condition: str, location: str | None
    ) -> RaceBenchmark | None: ...


@dataclass
class RaceBenchmarkRegistry:
    """In-memory registry; production is intentionally empty until configured."""

    benchmarks: list[RaceBenchmark] = field(default_factory=list)

    def get_race_benchmark(
        self, *, condition: str, location: str | None
    ) -> RaceBenchmark | None:
        condition_key = condition.strip().casefold()
        location_key = (location or "").strip().casefold()
        matches = [
            benchmark
            for benchmark in self.benchmarks
            if benchmark.is_valid()
            and (
                benchmark.condition_scope is None
                or benchmark.condition_scope.strip().casefold() == condition_key
            )
            and (
                benchmark.location_scope is None
                or benchmark.location_scope.strip().casefold() == location_key
            )
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda benchmark: (
                benchmark.condition_scope is not None,
                benchmark.location_scope is not None,
                benchmark.year or -1,
            ),
        )


DEFAULT_RACE_BENCHMARK_PROVIDER = RaceBenchmarkRegistry()
