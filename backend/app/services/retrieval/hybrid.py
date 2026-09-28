"""Fuses lexical + dense retrieval with structured preference signals into one score.

This module answers "how relevant is this trial to what the patient described",
nothing more. It never touches whether the patient structurally qualifies for a
trial (age/sex/recruitment) — that is a separate, deterministic judgment made by
``backend.app.services.eligibility`` and must stay independent of relevance so a
verbose or matching eligibility blurb can't buy a trial a higher rank.
"""
import math

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.retrieval.dense import DenseRetriever
from backend.app.services.retrieval.documents import trial_document
from backend.app.services.retrieval.lexical import BM25Retriever
from backend.app.services.text import tokenize_list

# Relative contribution of each signal to the final [0, 1] score. One place to
# tune the ranking formula; weights sum to 1.0.
WEIGHTS = {
    "lexical": 0.30,
    "semantic": 0.30,
    "condition": 0.20,
    "intervention": 0.08,
    "phase": 0.07,
    "location": 0.05,
}
if not math.isclose(sum(WEIGHTS.values()), 1.0):
    raise RuntimeError("Hybrid retrieval weights must sum to 1.0")


class HybridRetriever:
    """Ranks trials for a patient profile using BM25 + dense embeddings + structured signals."""

    def __init__(self) -> None:
        self._lexical = BM25Retriever()
        self._dense = DenseRetriever()

    def rank(
        self, patient: PatientProfile, trials: list[Trial]
    ) -> list[tuple[Trial, float, dict[str, float], dict[str, float], list[str]]]:
        """Return trial, overall, weighted, normalized components, and terms."""
        if not trials:
            return []

        query = self._profile_text(patient)
        query_tokens = tokenize_list(query)
        documents = [trial_document(trial) for trial in trials]
        document_tokens = [tokenize_list(document) for document in documents]
        lexical_scores = self._lexical.score(
            query,
            trials,
            query_tokens=query_tokens,
            document_tokens=document_tokens,
        )
        semantic_scores = self._dense.score(query, trials, documents=documents)
        query_terms = set(query_tokens)

        ranked = []
        for trial, document_terms, lexical_score, semantic_score in zip(
            trials, document_tokens, lexical_scores, semantic_scores
        ):
            components = self._component_scores(patient, trial, lexical_score, semantic_score)
            breakdown = {
                name: round(components[name] * weight, 4)
                for name, weight in WEIGHTS.items()
            }
            score = round(max(0.0, min(1.0, sum(breakdown.values()))), 4)
            overlap = sorted(query_terms.intersection(document_terms))
            ranked.append((trial, score, breakdown, components, overlap[:8]))

        return sorted(ranked, key=lambda item: (-item[1], item[0].nct_id))

    def _component_scores(
        self, patient: PatientProfile, trial: Trial, lexical_score: float, semantic_score: float
    ) -> dict[str, float]:
        condition = 1.0 if any(
            patient.condition.casefold() in value.casefold() for value in trial.conditions
        ) else 0.0
        wanted_phases = {value.strip().casefold() for value in patient.phase_preferences}
        trial_phases = {value.strip().casefold() for value in trial.phases}
        phase = 1.0 if wanted_phases.intersection(trial_phases) else 0.0
        intervention_text = " ".join(trial.interventions).casefold()
        intervention = 1.0 if any(
            preference.casefold() in intervention_text
            for preference in patient.intervention_preferences
        ) else 0.0
        location = 1.0 if (
            patient.location
            and patient.location.casefold() in " ".join(trial.locations).casefold()
        ) else 0.0
        return {
            "lexical": round(lexical_score, 4),
            "semantic": round(semantic_score, 4),
            "condition": condition,
            "intervention": intervention,
            "phase": phase,
            "location": location,
        }

    def _profile_text(self, patient: PatientProfile) -> str:
        return " ".join(
            part
            for part in [
                patient.condition,
                patient.notes or "",
            ]
            if part
        )
