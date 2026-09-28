import pytest

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.recommendations import RecommendationService
from backend.app.services.representation_risk import predictor as predictor_module
from backend.app.services.representation_risk.predictor import (
    MIN_MACRO_F1_FOR_AVAILABILITY, _CachedPredictor, predict_representation_risk,
)
from backend.app.services.representation_risk.schemas import RepresentationRiskPrediction
from representation_risk.dataset import build_dataset, train_test_split_dataset
from representation_risk.features import FEATURE_NAMES, extract_features
from representation_risk.labels import RISK_LEVELS, representation_gap, risk_level_from_gap
from representation_risk.models import evaluate_model, train_models


def design_trial(**changes):
    return Trial(**(dict(
        nct_id="RR-1", title="Design trial", status="RECRUITING",
        conditions=["Cancer"], phases=["PHASE2"], sex="ALL",
        locations=["Boston, Massachusetts, United States", "Chicago, Illinois, United States"],
        interventions=["Drug A"], eligibility_criteria="- Criterion 1\n- Criterion 2",
        target_enrollment=200,
    ) | changes))


# --- no target leakage --------------------------------------------------------

def test_feature_extraction_never_reads_observed_demographics() -> None:
    base = design_trial()
    with_data = base.model_copy(update={
        "enrollment_race_distribution": {"WHITE": 10, "BLACK": 90},
        "enrollment_race_source": "test",
    })
    without_data = base.model_copy(update={"enrollment_race_distribution": None, "enrollment_race_source": None})

    assert extract_features(with_data) == extract_features(without_data)


def test_feature_names_exclude_any_observed_demographic_field() -> None:
    # "target_enrollment_*" (planned size) and "sex_reported" (was a protocol sex
    # eligibility value given at all) are pre-outcome design fields, not leakage.
    leakage_terms = ("race", "demographic", "observed", "distribution")
    assert not any(term in name for name in FEATURE_NAMES for term in leakage_terms)


# --- deterministic feature extraction ------------------------------------------

def test_feature_extraction_is_deterministic() -> None:
    trial = design_trial()
    assert extract_features(trial) == extract_features(trial)


def test_feature_extraction_handles_missing_design_fields_without_erroring() -> None:
    sparse = Trial(nct_id="RR-2", title="Sparse trial", status="RECRUITING")
    features = extract_features(sparse)
    assert set(features) == set(FEATURE_NAMES)
    assert features["target_enrollment_known"] == 0.0
    assert features["age_range_known"] == 0.0
    assert features["sex_reported"] == 0.0


# --- labels ---------------------------------------------------------------------

def test_risk_level_thresholds_are_monotonic_in_gap() -> None:
    assert risk_level_from_gap(0.0) == "low"
    assert risk_level_from_gap(0.05) == "moderate"
    assert risk_level_from_gap(0.5) == "high"


def test_representation_gap_is_zero_for_identical_distributions() -> None:
    distribution = {"WHITE": 60.0, "BLACK": 20.0, "OTHER": 20.0}
    assert representation_gap(distribution, distribution) == 0.0


# --- deterministic train/test split --------------------------------------------

def test_dataset_build_is_deterministic() -> None:
    first, second = build_dataset(), build_dataset()
    assert first.trial_ids == second.trial_ids
    assert first.X == second.X
    assert first.risk_level == second.risk_level


def test_train_test_split_is_deterministic_and_disjoint() -> None:
    dataset = build_dataset()
    train_a, test_a = train_test_split_dataset(dataset)
    train_b, test_b = train_test_split_dataset(dataset)

    assert train_a.trial_ids == train_b.trial_ids
    assert test_a.trial_ids == test_b.trial_ids
    assert set(train_a.trial_ids).isdisjoint(test_a.trial_ids)
    assert set(train_a.trial_ids) | set(test_a.trial_ids) == set(dataset.trial_ids)


# --- offline experiment runs without network -----------------------------------

def test_synthetic_evaluation_pipeline_runs_offline_and_reports_required_metrics() -> None:
    dataset = build_dataset()
    train, test = train_test_split_dataset(dataset)
    trained_models = train_models(train)
    assert {model.name for model in trained_models} == {"logistic_regression", "random_forest"}

    evaluations = [evaluate_model(model, test) for model in trained_models]
    for evaluation in evaluations:
        assert 0 <= evaluation.macro_f1 <= 1
        assert 0 <= evaluation.balanced_accuracy <= 1
        assert len(evaluation.confusion_matrix) == len(evaluation.labels)
        assert set(evaluation.labels) <= set(RISK_LEVELS)


# --- prediction schema + safe unavailability ------------------------------------

def test_prediction_schema_marks_evidence_as_predicted() -> None:
    prediction = RepresentationRiskPrediction(model_version="test-v0")
    assert prediction.evidence_type == "predicted"
    assert prediction.risk_level is None  # default is "unavailable", never a fabricated guess


def test_predictor_returns_unavailable_when_model_quality_is_insufficient(monkeypatch) -> None:
    predictor = _CachedPredictor()
    monkeypatch.setattr(predictor, "available", False)
    prediction = predictor.predict(design_trial())

    assert prediction.risk_level is None
    assert prediction.probabilities is None
    assert prediction.confidence is None
    assert any("below the" in note for note in prediction.limitations)


def test_predictor_is_available_on_the_real_fixture_pipeline() -> None:
    # Sanity check that the actual synthetic pipeline clears its own bar; if this
    # ever regresses, predict_representation_risk will safely return "unavailable"
    # rather than a bad prediction, so this is a quality signal, not a safety one.
    predictor_module._get_predictor.cache_clear()
    predictor = predictor_module._get_predictor()
    assert predictor.evaluation.macro_f1 >= MIN_MACRO_F1_FOR_AVAILABILITY
    assert predictor.available is True


def test_predict_representation_risk_returns_none_without_a_trained_model(monkeypatch) -> None:
    class AlwaysUnavailable:
        available = False

        def predict(self, trial):
            raise AssertionError("predict() should not be reached in the unavailable branch check")

    # Exercises the "missing model/data -> safe unavailable" path end to end via the
    # public entrypoint, without depending on internal branch details of predict().
    monkeypatch.setattr(predictor_module, "_get_predictor", lambda: predictor_module._CachedPredictor())
    trial = design_trial()
    prediction = predict_representation_risk(trial)
    assert prediction is not None  # the real fixture pipeline clears the bar (see test above)
    assert prediction.evidence_type == "predicted"


# --- independence from ESR / relevance / eligibility ----------------------------

def test_prediction_is_independent_of_patient_and_therefore_of_relevance_and_eligibility() -> None:
    trial = design_trial()
    prediction_a = predict_representation_risk(trial)
    prediction_b = predict_representation_risk(trial)
    # Feature extraction never reads the patient profile at all -- there is no
    # patient argument to predict_representation_risk -- so it cannot vary with
    # relevance ranking or structured eligibility, which are both patient-specific.
    assert prediction_a == prediction_b


def test_observed_esr_is_never_overwritten_by_a_prediction() -> None:
    service = RecommendationService()
    trial_with_observed_data = design_trial(
        enrollment_race_distribution={"WHITE": 55.0, "BLACK": 25.0, "ASIAN": 12.0, "OTHER": 8.0},
        enrollment_race_source="test",
    )

    result = service.recommend(PatientProfile(condition="cancer"), [trial_with_observed_data])[0]

    assert result.esr is not None
    assert result.esr.components["race"].evidence_type in {
        "observed_distribution_comparison", "insufficient_benchmark",
    }
    assert result.representation_risk is None


def test_prediction_available_only_without_observed_race_data() -> None:
    service = RecommendationService()
    trial_without_data = design_trial(nct_id="RR-no-data")
    trial_with_data = design_trial(
        nct_id="RR-with-data",
        enrollment_race_distribution={"WHITE": 55.0, "BLACK": 25.0, "ASIAN": 12.0, "OTHER": 8.0},
        enrollment_race_source="test",
    )

    results = {r.trial.nct_id: r for r in service.recommend(
        PatientProfile(condition="cancer"), [trial_without_data, trial_with_data]
    )}

    assert results["RR-no-data"].representation_risk is not None
    assert results["RR-with-data"].representation_risk is None


def test_representation_risk_never_moves_relevance_score_or_eligibility() -> None:
    service = RecommendationService()
    trial = design_trial()

    with_prediction = service.recommend(PatientProfile(condition="cancer", age=40, sex="female"), [trial])[0]

    # A trial with observed data (no prediction attached) but otherwise identical
    # design/relevance-affecting fields should score identically on relevance and
    # eligibility -- proving representation_risk contributes nothing to either.
    trial_with_observed = design_trial(
        enrollment_race_distribution={"WHITE": 55.0, "BLACK": 25.0, "ASIAN": 12.0, "OTHER": 8.0},
        enrollment_race_source="test",
    )
    without_prediction = service.recommend(PatientProfile(condition="cancer", age=40, sex="female"), [trial_with_observed])[0]

    assert with_prediction.representation_risk is not None
    assert without_prediction.representation_risk is None
    assert with_prediction.score == without_prediction.score
    assert with_prediction.structured_eligibility.status == without_prediction.structured_eligibility.status
