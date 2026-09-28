"""Two lightweight sklearn baselines for representation risk level. No deep learning.

Neither is hard-coded as "the winner" here -- representation_risk/run.py reports
both models' metrics side by side; representation_risk/predictor.py (the backend
serving layer) picks between them by measured macro F1 on the held-out split, not
by a fixed preference.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from representation_risk.dataset import Dataset
from representation_risk.features import FEATURE_NAMES
from representation_risk.labels import RISK_LEVELS

MODEL_VERSION = "representation-risk-experimental-v1"


@dataclass
class TrainedModel:
    name: str
    model: object
    scaler: StandardScaler | None
    feature_importance: dict[str, float]


@dataclass
class EvaluationResult:
    name: str
    macro_f1: float
    balanced_accuracy: float
    confusion_matrix: list[list[int]]
    labels: list[str]


def _fit_logistic_regression(X: np.ndarray, y: list[str]) -> TrainedModel:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    # Mean absolute coefficient across the one-vs-rest classes: a simple, fully
    # inspectable importance proxy -- no SHAP needed for a model this small.
    importance = np.abs(model.coef_).mean(axis=0)
    return TrainedModel(
        name="logistic_regression",
        model=model,
        scaler=scaler,
        feature_importance=dict(zip(FEATURE_NAMES, importance.tolist())),
    )


def _fit_random_forest(X: np.ndarray, y: list[str]) -> TrainedModel:
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X, y)
    return TrainedModel(
        name="random_forest",
        model=model,
        scaler=None,
        feature_importance=dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())),
    )


def train_models(train: Dataset) -> list[TrainedModel]:
    X = np.array(train.X)
    return [_fit_logistic_regression(X, train.risk_level), _fit_random_forest(X, train.risk_level)]


def predict_labels(trained: TrainedModel, X: np.ndarray) -> list[str]:
    inputs = trained.scaler.transform(X) if trained.scaler is not None else X
    return list(trained.model.predict(inputs))


def evaluate_model(trained: TrainedModel, test: Dataset) -> EvaluationResult:
    X = np.array(test.X)
    predictions = predict_labels(trained, X)
    labels = [level for level in RISK_LEVELS if level in set(test.risk_level) | set(predictions)]
    return EvaluationResult(
        name=trained.name,
        macro_f1=round(float(f1_score(test.risk_level, predictions, average="macro", zero_division=0)), 4),
        balanced_accuracy=round(float(balanced_accuracy_score(test.risk_level, predictions)), 4),
        confusion_matrix=confusion_matrix(test.risk_level, predictions, labels=labels).tolist(),
        labels=labels,
    )
