"""Backend-facing serving layer for the EXPERIMENTAL representation-risk model.

Trains once per process (on synthetic fixture data -- see
representation_risk/README.md) and caches the result, the same pattern
`backend.app.services.retrieval.dense` uses for its embedding model. Never
presented as validated real-world predictive performance.
"""
from functools import lru_cache

import numpy as np

from backend.app.models.schemas import Trial
from backend.app.services.representation_risk.schemas import RepresentationRiskPrediction
from representation_risk.dataset import build_dataset, train_test_split_dataset
from representation_risk.features import FEATURE_NAMES, extract_features
from representation_risk.models import MODEL_VERSION, evaluate_model, predict_labels, train_models

# Roughly chance level for a balanced 3-class problem (1/3 ~= 0.333); below this,
# the fixture evaluation says the model isn't doing better than guessing, so
# predictions are withheld rather than shown with false confidence.
MIN_MACRO_F1_FOR_AVAILABILITY = 0.34
TOP_DRIVER_COUNT = 3

LIMITATIONS = [
    "EXPERIMENTAL and NOT VALIDATED: trained and evaluated on synthetic fixture data only, not real historical trials.",
    "Predicts design/access-driven risk only; it has no information about this specific trial's actual enrollment.",
    "Never a substitute for observed ESR evidence when real enrollment demographics are available.",
]


class _CachedPredictor:
    def __init__(self) -> None:
        dataset = build_dataset()
        train, test = train_test_split_dataset(dataset)
        self.trained_models = train_models(train)
        evaluations = {model.name: evaluate_model(model, test) for model in self.trained_models}
        # Selected by measured macro F1 on the held-out fixture split, not a fixed
        # preference; ties broken alphabetically for determinism.
        best_name = max(evaluations, key=lambda name: (evaluations[name].macro_f1, name))
        self.model = next(model for model in self.trained_models if model.name == best_name)
        self.evaluation = evaluations[best_name]
        self.available = self.evaluation.macro_f1 >= MIN_MACRO_F1_FOR_AVAILABILITY

    def predict(self, trial: Trial) -> RepresentationRiskPrediction:
        if not self.available:
            return RepresentationRiskPrediction(
                risk_level=None,
                probabilities=None,
                confidence=None,
                model_version=MODEL_VERSION,
                drivers=[],
                limitations=LIMITATIONS + [
                    f"Selected model ({self.model.name}) scored macro F1 {self.evaluation.macro_f1:.2f} "
                    f"on the fixture evaluation, below the {MIN_MACRO_F1_FOR_AVAILABILITY:.2f} "
                    "availability threshold, so no prediction is returned."
                ],
            )

        features = extract_features(trial)
        X = np.array([[features[name] for name in FEATURE_NAMES]])
        predicted = predict_labels(self.model, X)[0]

        probabilities = None
        confidence = None
        if hasattr(self.model.model, "predict_proba"):
            inputs = self.model.scaler.transform(X) if self.model.scaler is not None else X
            proba = self.model.model.predict_proba(inputs)[0]
            probabilities = {label: round(float(p), 4) for label, p in zip(self.model.model.classes_, proba)}
            confidence = round(float(max(proba)), 4)

        top_drivers = sorted(self.model.feature_importance.items(), key=lambda item: -item[1])[:TOP_DRIVER_COUNT]
        drivers = [f"{name} (importance {value:.3f})" for name, value in top_drivers]

        return RepresentationRiskPrediction(
            risk_level=predicted,
            probabilities=probabilities,
            confidence=confidence,
            model_version=MODEL_VERSION,
            drivers=drivers,
            limitations=LIMITATIONS,
        )


@lru_cache(maxsize=1)
def _get_predictor() -> _CachedPredictor:
    return _CachedPredictor()


def predict_representation_risk(trial: Trial) -> RepresentationRiskPrediction | None:
    """Returns a prediction only for trials without observed race enrollment data.

    Trials with real reported demographics already have authoritative ESR
    evidence (backend.app.services.equity); returning a prediction alongside
    that would compete with, not complement, it. Callers should surface ESR
    for those trials and treat a None here as "no prediction to show."
    """
    if trial.enrollment_race_distribution:
        return None
    return _get_predictor().predict(trial)
