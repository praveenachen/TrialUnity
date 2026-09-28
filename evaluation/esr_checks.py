"""Lightweight ESR validation, deliberately separate from retrieval quality metrics.

ESR is not a relevance signal, so it is never scored against the benchmark's
relevance grades, and its "quality" is not an IR question -- these checks only
prove the structural guarantees ESR is supposed to hold (see evaluation/README.md
for why IR quality and ESR validation are kept apart).
"""
from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.equity.service import compute_esr
from backend.app.services.retrieval.hybrid import HybridRetriever


def esr_is_deterministic(patient: PatientProfile, trial: Trial) -> bool:
    return compute_esr(patient, trial) == compute_esr(patient, trial)


def missing_evidence_changes_coverage_not_score(patient: PatientProfile, trial: Trial) -> dict:
    """Removing sex enrollment evidence should lower coverage, not fabricate a 0 score."""
    with_data = trial.model_copy(update={
        "enrollment_sex_distribution": {"MALE": 40, "FEMALE": 60},
        "enrollment_sex_source": "evaluation-fixture",
    })
    without_data = trial.model_copy(update={
        "enrollment_sex_distribution": None,
        "enrollment_sex_source": None,
    })

    esr_with = compute_esr(patient, with_data)
    esr_without = compute_esr(patient, without_data)

    return {
        "coverage_with_evidence": esr_with.evidence_coverage,
        "coverage_without_evidence": esr_without.evidence_coverage,
        "coverage_decreased": esr_without.evidence_coverage < esr_with.evidence_coverage,
        "score_not_forced_to_zero": esr_without.score is None or esr_without.score > 0,
    }


def esr_does_not_change_clinical_ordering(patient: PatientProfile, trial_a: Trial, trial_b: Trial) -> bool:
    """Varying only ESR-relevant fields (never read by retrieval) must not reorder relevance."""
    baseline_order = [trial.nct_id for trial, *_ in HybridRetriever().rank(patient, [trial_a, trial_b])]

    enriched_a = trial_a.model_copy(update={
        "enrollment_sex_distribution": {"MALE": 90, "FEMALE": 10},
        "enrollment_sex_source": "evaluation-fixture",
        "enrollment_race_distribution": {"WHITE": 95, "BLACK": 5},
        "enrollment_race_source": "evaluation-fixture",
    })
    enriched_order = [trial.nct_id for trial, *_ in HybridRetriever().rank(patient, [enriched_a, trial_b])]

    return baseline_order == enriched_order


def run_esr_checks() -> dict:
    patient = PatientProfile(condition="cancer", location="Boston")
    trial_a = Trial(
        nct_id="ESR-CHECK-A", title="Cancer trial A", status="RECRUITING",
        conditions=["Cancer"], locations=["Boston, Massachusetts, United States"],
    )
    trial_b = Trial(
        nct_id="ESR-CHECK-B", title="Cancer trial B", status="RECRUITING",
        conditions=["Cancer"], locations=["Chicago, Illinois, United States"],
    )

    return {
        "deterministic": esr_is_deterministic(patient, trial_a),
        "missing_evidence": missing_evidence_changes_coverage_not_score(patient, trial_a),
        "does_not_change_clinical_ordering": esr_does_not_change_clinical_ordering(patient, trial_a, trial_b),
    }
