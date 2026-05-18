from backend.app.models.schemas import MatchExplanation, PatientProfile, Trial, TrialRecommendation
from backend.app.services.retrieval import SemanticRetriever
from backend.app.services.text import first_sentence


class RecommendationService:
    def __init__(self) -> None:
        self.retriever = SemanticRetriever()

    def recommend(self, patient: PatientProfile, trials: list[Trial], limit: int = 10) -> list[TrialRecommendation]:
        ranked = self.retriever.rank(patient, trials)
        return [
            TrialRecommendation(
                trial=trial,
                score=score,
                explanation=self._explain(patient, trial, score, matched_terms),
            )
            for trial, score, matched_terms in ranked[:limit]
        ]

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
            f"Semantic relevance score {score:.2f}",
            f"condition focus includes {', '.join(trial.conditions[:3]) or 'related clinical terms'}",
        ]
        if matched_terms:
            rationale_bits.append(f"shared terms: {', '.join(matched_terms[:5])}")

        return MatchExplanation(
            matched_terms=matched_terms,
            eligibility_notes=eligibility_notes,
            ranking_rationale="; ".join(rationale_bits) + ".",
            patient_friendly_summary=first_sentence(trial.brief_summary),
        )
