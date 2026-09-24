from backend.app.models.schemas import MatchExplanation, PatientProfile, Trial, TrialRecommendation
from backend.app.services.eligibility import evaluate_eligibility
from backend.app.services.retrieval import SemanticRetriever
from backend.app.services.text import first_sentence


class RecommendationService:
    def __init__(self) -> None:
        self.retriever = SemanticRetriever()

    def recommend(self, patient: PatientProfile, trials: list[Trial], limit: int = 10) -> list[TrialRecommendation]:
        ranked = self.retriever.rank(patient, trials)
        results = []
        for trial, score, matched_terms in ranked:
            eligibility = evaluate_eligibility(patient, trial)
            explanation = self._explain(patient, trial, score, matched_terms)
            explanation.eligibility_notes.extend(
                criterion.reason for criterion in eligibility.criteria.values()
                if criterion.state == "incompatible"
            )
            explanation.manual_review_signals = [
                criterion.reason for criterion in eligibility.criteria.values()
                if criterion.state == "unknown"
            ] + ["Full eligibility criteria and site availability require study-team review."]
            results.append(TrialRecommendation(
                trial=trial, score=score, explanation=explanation,
                structured_eligibility=eligibility,
            ))
        # Keep relevance unchanged; known structured conflicts form a separate final group.
        results.sort(key=lambda result: result.structured_eligibility.status == "incompatible")
        return results[:limit]

    def _explain(self, patient: PatientProfile, trial: Trial, score: float, matched_terms: list[str]) -> MatchExplanation:
        eligibility_notes = []
        if trial.sex and trial.sex.lower() != "all":
            eligibility_notes.append(f"Sex listed by the study: {trial.sex}.")
        if trial.minimum_age or trial.maximum_age:
            eligibility_notes.append(
                f"Age range listed by the study: {trial.minimum_age or 'not specified'} to {trial.maximum_age or 'not specified'}."
            )
        if trial.eligibility_criteria:
            eligibility_notes.append("Eligibility criteria should be reviewed with the study team before outreach.")

        rationale_bits = [
            f"Relative relevance score {score:.2f} (not medical eligibility)",
            f"condition focus includes {', '.join(trial.conditions[:3]) or 'related clinical terms'}",
        ]
        if matched_terms:
            rationale_bits.append(f"shared terms: {', '.join(matched_terms[:5])}")

        relevant_signals = []
        if any(patient.condition.lower() in condition.lower() for condition in trial.conditions):
            relevant_signals.append("Condition text matches.")
        if set(patient.phase_preferences).intersection(trial.phases):
            relevant_signals.append("Preferred phase matches.")
        if any(pref.lower() in " ".join(trial.interventions).lower() for pref in patient.intervention_preferences):
            relevant_signals.append("Preferred intervention text matches.")
        if patient.location and patient.location.lower() in " ".join(trial.locations).lower():
            relevant_signals.append("Location text matches.")

        return MatchExplanation(
            matched_terms=matched_terms,
            relevant_signals=relevant_signals,
            eligibility_notes=eligibility_notes,
            ranking_rationale="; ".join(rationale_bits) + ".",
            patient_friendly_summary=first_sentence(trial.brief_summary),
        )
