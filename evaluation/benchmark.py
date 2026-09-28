"""Loads the curated retrieval benchmark from evaluation/fixtures/benchmark.json.

All patients and trials are synthetic (invented condition/trial names for this
benchmark) -- nothing here is a real patient record or a real registered trial.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from backend.app.models.schemas import PatientProfile, Trial

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "benchmark.json"


@dataclass
class BenchmarkCase:
    id: str
    category: str
    description: str
    patient: PatientProfile
    graded_trials: list[tuple[Trial, int]]

    @property
    def trials(self) -> list[Trial]:
        return [trial for trial, _grade in self.graded_trials]

    @property
    def relevance_by_id(self) -> dict[str, int]:
        return {trial.nct_id: grade for trial, grade in self.graded_trials}


def load_benchmark(path: Path = FIXTURE_PATH) -> list[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for raw in payload["cases"]:
        graded_trials = [(Trial(**entry["trial"]), entry["relevance"]) for entry in raw["trials"]]
        cases.append(BenchmarkCase(
            id=raw["id"],
            category=raw["category"],
            description=raw.get("description", ""),
            patient=PatientProfile(**raw["patient"]),
            graded_trials=graded_trials,
        ))
    return cases
