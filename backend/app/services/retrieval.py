from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.text import tokenize


class SemanticRetriever:
    def rank(self, patient: PatientProfile, trials: list[Trial]) -> list[tuple[Trial, float, list[str]]]:
        if not trials:
            return []

        query = self._profile_text(patient)
        documents = [self._trial_text(trial) for trial in trials]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([query, *documents])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        query_terms = tokenize(query)

        ranked = []
        for trial, semantic_score in zip(trials, similarities):
            overlap = sorted(query_terms.intersection(tokenize(self._trial_text(trial))))
            score = self._weighted_score(patient, trial, float(semantic_score), overlap)
            ranked.append((trial, score, overlap[:8]))

        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _weighted_score(self, patient: PatientProfile, trial: Trial, semantic_score: float, overlap: list[str]) -> float:
        condition_match = 0.2 if any(patient.condition.lower() in c.lower() for c in trial.conditions) else 0
        phase_match = 0.08 if patient.phase_preferences and set(patient.phase_preferences).intersection(trial.phases) else 0
        intervention_match = 0.08 if any(
            pref.lower() in " ".join(trial.interventions).lower() for pref in patient.intervention_preferences
        ) else 0
        location_match = 0.07 if patient.location and patient.location.lower() in " ".join(trial.locations).lower() else 0
        explainability_bonus = min(len(overlap) * 0.015, 0.12)
        score = (semantic_score * 0.55) + condition_match + phase_match + intervention_match + location_match + explainability_bonus
        return round(max(0, min(score, 1)), 3)

    def _profile_text(self, patient: PatientProfile) -> str:
        return " ".join(
            part
            for part in [
                patient.condition,
                patient.sex or "",
                patient.location or "",
                " ".join(patient.intervention_preferences),
                " ".join(patient.phase_preferences),
                patient.notes or "",
            ]
            if part
        )

    def _trial_text(self, trial: Trial) -> str:
        return " ".join(
            part
            for part in [
                trial.title,
                " ".join(trial.conditions),
                " ".join(trial.interventions),
                " ".join(trial.phases),
                trial.brief_summary or "",
                trial.eligibility_criteria or "",
                " ".join(trial.locations),
            ]
            if part
        )
