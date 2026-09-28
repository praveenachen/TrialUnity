import pytest

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.equity.race import RaceBenchmark, observed_race_component
from backend.app.services.equity.service import WEIGHTS, compute_esr
from backend.app.services.equity.sex import observed_sex_component, prospective_sex_component
from backend.app.services.equity.socioeconomic import socioeconomic_component
from backend.app.services.recommendations import RecommendationService


def trial(**changes):
    return Trial(**(dict(
        nct_id="NCT1", title="Cancer study", status="RECRUITING",
        conditions=["Cancer"], sex="ALL",
        locations=["Boston, Massachusetts, United States", "Toronto, Ontario, Canada"],
    ) | changes))


RACE_BENCHMARK = RaceBenchmark(
    distribution={"WHITE": 0.6, "BLACK": 0.2, "ASIAN": 0.1, "OTHER": 0.1},
    source="Test census reference distribution",
)


def test_full_esr_with_all_components_available() -> None:
    patient = PatientProfile(condition="cancer", location="Boston")
    subject = trial(
        enrollment_sex_distribution={"MALE": 48, "FEMALE": 52},
        enrollment_race_distribution={"WHITE": 55, "BLACK": 25, "ASIAN": 12, "OTHER": 8},
    )

    result = compute_esr(patient, subject, race_benchmark=RACE_BENCHMARK)

    assert result.mode == "observed"
    assert result.score is not None
    assert result.evidence_coverage == 1.0
    assert all(component.score is not None for component in result.components.values())


def test_missing_race_data_reduces_coverage_but_not_score_to_zero() -> None:
    patient = PatientProfile(condition="cancer", location="Boston")
    subject = trial(enrollment_sex_distribution={"MALE": 50, "FEMALE": 50})

    result = compute_esr(patient, subject)  # no race benchmark, no race enrollment

    assert result.components["race"].score is None
    assert result.components["race"].evidence_type == "insufficient_data"
    assert result.score is not None  # still computed from the other two components
    assert result.evidence_coverage < 1.0
    assert result.mode in {"observed", "mixed"}


def test_missing_socioeconomic_evidence_does_not_zero_the_score() -> None:
    patient = PatientProfile(condition="cancer")  # no location
    subject = trial(locations=[])  # no sites

    component = socioeconomic_component(patient, subject)

    assert component.score is None
    assert component.evidence_coverage == 0.0
    assert component.evidence_type == "insufficient_data"
    assert len(component.missing_evidence) == 2


def test_missing_sex_evidence_falls_back_to_insufficient_data() -> None:
    subject = trial(sex=None)
    assert observed_sex_component(subject) is None

    component = prospective_sex_component(subject)
    assert component.score is None
    assert component.evidence_type == "insufficient_data"


def test_no_evidence_at_all_is_insufficient_data() -> None:
    patient = PatientProfile(condition="cancer")
    subject = trial(sex=None, locations=[])

    result = compute_esr(patient, subject)

    assert result.score is None
    assert result.mode == "insufficient_data"
    assert result.evidence_coverage == 0.0


def test_weight_normalization_when_components_are_missing() -> None:
    # Only socioeconomic (0.35) and sex (0.25) available; race (0.40) unavailable.
    # Score should be the weighted average of the two available components only.
    patient = PatientProfile(condition="cancer", location="Boston")
    subject = trial()  # ALL sex (100), full site/proximity signals, no race data

    result = compute_esr(patient, subject)
    socioeconomic_score = result.components["socioeconomic"].score
    sex_score = result.components["sex"].score
    assert result.components["race"].score is None

    expected = round(
        (WEIGHTS["socioeconomic"] * socioeconomic_score + WEIGHTS["sex"] * sex_score)
        / (WEIGHTS["socioeconomic"] + WEIGHTS["sex"]),
        2,
    )
    assert result.score == expected


def test_missing_evidence_never_becomes_zero_or_hundred() -> None:
    patient = PatientProfile(condition="cancer")
    subject = trial(sex=None, locations=[])
    result = compute_esr(patient, subject)

    for component in result.components.values():
        if component.evidence_type in {"insufficient_data", "insufficient_benchmark"}:
            assert component.score is None


def test_prospective_sex_inclusivity_is_not_observed_representation() -> None:
    all_sexes = prospective_sex_component(trial(sex="ALL"))
    single_sex = prospective_sex_component(trial(sex="FEMALE"))

    assert all_sexes.evidence_type == "protocol_inclusivity"
    assert single_sex.evidence_type == "protocol_inclusivity"
    assert "PROSPECTIVE" in all_sexes.rationale
    assert all_sexes.score == 100
    assert single_sex.score == 0

    # Reported enrollment data must take precedence and be labeled differently.
    observed = observed_sex_component(trial(enrollment_sex_distribution={"MALE": 50, "FEMALE": 50}))
    assert observed.evidence_type == "observed_enrollment"
    assert "OBSERVED" in observed.rationale


def test_race_scoring_requires_both_observed_distribution_and_benchmark() -> None:
    no_enrollment = observed_race_component(trial(), RACE_BENCHMARK)
    assert no_enrollment.score is None
    assert no_enrollment.evidence_type == "insufficient_data"

    no_benchmark = observed_race_component(
        trial(enrollment_race_distribution={"WHITE": 10, "BLACK": 10}), None
    )
    assert no_benchmark.score is None
    assert no_benchmark.evidence_type == "insufficient_benchmark"

    both = observed_race_component(
        trial(enrollment_race_distribution={"WHITE": 60, "BLACK": 20, "ASIAN": 10, "OTHER": 10}),
        RACE_BENCHMARK,
    )
    assert both.score == 100.0  # identical distribution to the benchmark
    assert both.evidence_type == "observed_distribution_comparison"
    assert both.source == RACE_BENCHMARK.source


def test_race_similarity_decreases_as_distributions_diverge() -> None:
    identical = observed_race_component(
        trial(enrollment_race_distribution={"WHITE": 60, "BLACK": 20, "ASIAN": 10, "OTHER": 10}),
        RACE_BENCHMARK,
    )
    skewed = observed_race_component(
        trial(enrollment_race_distribution={"WHITE": 95, "BLACK": 1, "ASIAN": 2, "OTHER": 2}),
        RACE_BENCHMARK,
    )
    assert identical.score > skewed.score


def test_deterministic_esr_output() -> None:
    patient = PatientProfile(condition="cancer", location="Boston")
    subject = trial(
        enrollment_sex_distribution={"MALE": 40, "FEMALE": 60},
        enrollment_race_distribution={"WHITE": 55, "BLACK": 25, "ASIAN": 12, "OTHER": 8},
    )

    first = compute_esr(patient, subject, race_benchmark=RACE_BENCHMARK)
    second = compute_esr(patient, subject, race_benchmark=RACE_BENCHMARK)

    assert first == second


def test_esr_is_independent_of_relevance_and_eligibility() -> None:
    service = RecommendationService()
    subject = trial(minimum_age="18 Years", maximum_age="99 Years")

    # Age changes structured eligibility (and nothing about ESR's evidence sources).
    eligible = service.recommend(PatientProfile(condition="cancer", age=40, sex="female", location="Boston"), [subject])[0]
    ineligible = service.recommend(PatientProfile(condition="cancer", age=5, sex="female", location="Boston"), [subject])[0]
    assert eligible.structured_eligibility.status == "compatible"
    assert ineligible.structured_eligibility.status == "incompatible"
    assert eligible.esr == ineligible.esr
    assert eligible.score == ineligible.score  # age doesn't affect relevance either

    # Notes text changes lexical/semantic relevance but must not move ESR.
    verbose = service.recommend(
        PatientProfile(condition="cancer", age=40, sex="female", location="Boston", notes="stage IV metastatic recurrent"),
        [subject],
    )[0]
    assert verbose.esr == eligible.esr
    assert verbose.score != eligible.score


@pytest.mark.parametrize(
    ("patient_location", "trial_locations", "expected_evidence_type"),
    [
        (None, [], "insufficient_data"),
        ("Boston", [], "observed_geographic"),
        (None, ["Boston, Massachusetts, United States"], "observed_geographic"),
    ],
)
def test_socioeconomic_evidence_type_reflects_available_signals(
    patient_location, trial_locations, expected_evidence_type
) -> None:
    component = socioeconomic_component(
        PatientProfile(condition="cancer", location=patient_location),
        trial(locations=trial_locations),
    )
    assert component.evidence_type == expected_evidence_type
