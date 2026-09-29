"""Truthful retrieval-funnel counts for the client's matching-progress UI.

This module only counts; it never re-ranks, re-scores, or filters candidates.
`build_funnel` is meant to be called with the *full* ranked list (i.e.
`RecommendationService.recommend(..., limit=len(trials))`) before the caller
truncates it to the requested page size, so `structured_eligible_trials` reflects
every candidate actually evaluated, not just the ones that made the final cut.
"""
from backend.app.models.schemas import MatchingFunnel, Trial, TrialRecommendation


def build_funnel(trials: list[Trial], ranked: list[TrialRecommendation], returned: int) -> MatchingFunnel:
    recruiting = sum(1 for trial in trials if trial.status.strip().upper() == "RECRUITING")
    structured_eligible = sum(
        1
        for result in ranked
        if result.structured_eligibility is None or result.structured_eligibility.status != "incompatible"
    )
    return MatchingFunnel(
        candidate_trials=len(trials),
        recruiting_trials=recruiting,
        structured_eligible_trials=structured_eligible,
        ranked_matches=returned,
    )
