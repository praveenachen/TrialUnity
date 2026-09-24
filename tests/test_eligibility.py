import pytest

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.eligibility import evaluate_eligibility
from backend.app.services.recommendations import RecommendationService


def trial(**changes):
    return Trial(**(dict(nct_id="NCT1", title="Cancer study", minimum_age="18 Years",
                        maximum_age="65 Years", sex="ALL", status="RECRUITING") | changes))


@pytest.mark.parametrize("age,minimum,maximum,status", [
    (17, "incompatible", "compatible", "incompatible"),
    (66, "compatible", "incompatible", "incompatible"),
    (40, "compatible", "compatible", "compatible"),
    (18, "compatible", "compatible", "compatible"),
    (65, "compatible", "compatible", "compatible"),
    (None, "unknown", "unknown", "unknown"),
])
def test_age(age, minimum, maximum, status):
    result = evaluate_eligibility(PatientProfile(condition="cancer", age=age, sex="female"), trial())
    assert result.criteria["minimum_age"].state == minimum
    assert result.criteria["maximum_age"].state == maximum
    assert result.status == status


@pytest.mark.parametrize("bound,state", [(None, "unknown"), ("", "unknown"),
    ("6 Months", "unknown"), ("invalid", "unknown"), ("N/A", "compatible")])
def test_missing_and_unsupported_bounds(bound, state):
    result = evaluate_eligibility(PatientProfile(condition="cancer", age=40, sex="female"),
                                  trial(minimum_age=bound, maximum_age=bound))
    assert result.criteria["minimum_age"].state == state
    assert result.criteria["maximum_age"].state == state


@pytest.mark.parametrize("patient_sex,trial_sex,state", [
    ("female", "FEMALE", "compatible"), ("male", "ALL", "compatible"),
    ("female", "MALE", "incompatible"), (None, "ALL", "unknown"),
    ("female", None, "unknown"), ("other", "ALL", "unknown"),
    ("female", "unknown", "unknown"),
])
def test_sex(patient_sex, trial_sex, state):
    result = evaluate_eligibility(PatientProfile(condition="cancer", age=40, sex=patient_sex), trial(sex=trial_sex))
    assert result.criteria["sex"].state == state


@pytest.mark.parametrize("status,state", [("RECRUITING", "compatible"),
    ("COMPLETED", "incompatible"), ("ACTIVE_NOT_RECRUITING", "incompatible"),
    ("NOT_YET_RECRUITING", "unknown"), ("Unknown", "unknown"),
    ("ENROLLING_BY_INVITATION", "unknown")])
def test_recruitment(status, state):
    result = evaluate_eligibility(PatientProfile(condition="cancer"), trial(status=status))
    assert result.criteria["recruitment_status"].state == state


def test_incompatibility_is_separate_from_score_and_affects_order():
    service = RecommendationService()
    trials = [trial(nct_id="conflict", minimum_age="60 Years"), trial(nct_id="compatible")]
    results = service.recommend(PatientProfile(condition="cancer", age=40, sex="female"), trials)
    assert [r.trial.nct_id for r in results] == ["compatible", "conflict"]
    assert results[0].score == results[1].score
    assert results[1].structured_eligibility.status == "incompatible"
    assert any("minimum age 60" in note for note in results[1].explanation.eligibility_notes)
    assert results[0].explanation.manual_review_signals


def test_unknown_and_free_text_never_establish_eligibility():
    result = RecommendationService().recommend(PatientProfile(condition="cancer"), [
        trial(minimum_age=None, maximum_age=None, sex=None, eligibility_criteria="Everyone is eligible.")])[0]
    assert result.structured_eligibility.status == "unknown"
    assert len(result.explanation.manual_review_signals) >= 4


def test_contradictory_bounds():
    result = evaluate_eligibility(PatientProfile(condition="cancer", age=40),
                                  trial(minimum_age="80 Years", maximum_age="18 Years"))
    assert result.criteria["minimum_age"].state == "unknown"
    assert result.criteria["maximum_age"].state == "unknown"


def test_empty_vocabulary_and_no_trials():
    service = RecommendationService()
    patient = PatientProfile(condition="the")
    assert service.recommend(patient, []) == []
    assert service.recommend(patient, [Trial(nct_id="1", title="the")])[0].score == 0
