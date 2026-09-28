from evaluation.ablation import evaluate_case, run_ablation
from evaluation.benchmark import BenchmarkCase, load_benchmark
from evaluation.esr_checks import (
    esr_does_not_change_clinical_ordering,
    esr_is_deterministic,
    missing_evidence_changes_coverage_not_score,
    run_esr_checks,
)
from evaluation.metrics import dcg_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from backend.app.models.schemas import PatientProfile, Trial

REQUIRED_CATEGORIES = {
    "exact_condition",
    "related_terminology_low_lexical_overlap",
    "treatment_preference",
    "location_preference",
    "phase_preference",
    "eligibility_incompatibility",
    "distractor",
}


# --- metrics -----------------------------------------------------------------

def test_recall_at_k_counts_relevant_hits_within_the_cutoff() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"c", "d", "z"}  # "z" is not even a candidate
    assert recall_at_k(ranked, relevant, k=2) == 0.0
    assert recall_at_k(ranked, relevant, k=3) == round(1 / 3, 6)
    assert recall_at_k(ranked, relevant, k=4) == round(2 / 3, 6)
    assert recall_at_k(ranked, set(), k=4) == 0.0


def test_reciprocal_rank_is_inverse_of_first_relevant_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_is_one_for_the_ideal_ordering() -> None:
    assert ndcg_at_k([2, 1, 0], k=3) == 1.0
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0  # no relevance anywhere -> defined as 0, not divide-by-zero


def test_ndcg_penalizes_a_worse_ordering_of_the_same_relevances() -> None:
    ideal = ndcg_at_k([2, 1, 0], k=3)
    worse = ndcg_at_k([0, 1, 2], k=3)
    assert worse < ideal


def test_dcg_matches_a_hand_computed_example() -> None:
    # (2^rel - 1) / log2(rank + 1) summed over ranks 1..6, relevances 3,2,3,0,1,2:
    # 7/1 + 3/log2(3) + 7/log2(4) + 0 + 1/log2(6) + 3/log2(7) = 13.8483 (rounded).
    relevances = [3, 2, 3, 0, 1, 2]
    assert round(dcg_at_k(relevances, k=6), 3) == 13.848


# --- benchmark shape -----------------------------------------------------------

def test_benchmark_has_expected_size_and_category_coverage() -> None:
    cases = load_benchmark()
    assert 12 <= len(cases) <= 20
    categories = {case.category for case in cases}
    assert REQUIRED_CATEGORIES.issubset(categories)


def test_every_case_has_at_least_one_relevant_and_one_irrelevant_candidate() -> None:
    for case in load_benchmark():
        grades = {grade for _trial, grade in case.graded_trials}
        assert max(grades) >= 1, f"{case.id} has no relevant candidate"
        assert min(grades) == 0, f"{case.id} has no distractor"


def test_case_ids_and_trial_ids_are_unique() -> None:
    cases = load_benchmark()
    assert len({case.id for case in cases}) == len(cases)
    all_trial_ids = [trial.nct_id for case in cases for trial in case.trials]
    assert len(all_trial_ids) == len(set(all_trial_ids))


# --- ablation runner -----------------------------------------------------------

def test_ablation_runs_over_the_whole_benchmark_without_network_access() -> None:
    result = run_ablation()
    assert result["num_cases"] == len(load_benchmark())
    assert set(result["aggregate"]) == {"lexical", "dense", "hybrid"}
    for system_metrics in result["aggregate"].values():
        assert set(system_metrics) == {"mrr", "recall@3", "ndcg@3", "recall@5", "ndcg@5"}
        assert all(0 <= value <= 1 for value in system_metrics.values())


def test_ablation_is_deterministic() -> None:
    first = run_ablation()
    second = run_ablation()
    assert first["aggregate"] == second["aggregate"]


def test_case_diagnostics_are_compact_and_carry_expected_relevance() -> None:
    case = load_benchmark()[0]
    result = evaluate_case(case)
    diagnostics = result["diagnostics"]
    assert diagnostics["expected_relevant"]
    assert len(diagnostics["hybrid_ranking"]) == len(case.trials)
    for row in diagnostics["hybrid_ranking"]:
        assert {"nct_id", "score", "lexical_score", "semantic_score", "structured_contribution", "relevance"} <= row.keys()


def test_hybrid_beats_irrelevant_distractors_on_the_distractor_heavy_case() -> None:
    case = next(c for c in load_benchmark() if c.id == "distractor-heavy-diabetes")
    result = evaluate_case(case)
    top_id = result["diagnostics"]["hybrid_ranking"][0]["nct_id"]
    assert case.relevance_by_id[top_id] == 2


# --- ESR boundary: never mixed into relevance metrics --------------------------

def test_esr_is_deterministic() -> None:
    patient = PatientProfile(condition="cancer")
    trial = Trial(nct_id="X", title="Cancer trial", status="RECRUITING", conditions=["Cancer"])
    assert esr_is_deterministic(patient, trial)


def test_missing_esr_evidence_changes_coverage_not_score() -> None:
    patient = PatientProfile(condition="cancer")
    trial = Trial(nct_id="X", title="Cancer trial", status="RECRUITING", conditions=["Cancer"])
    result = missing_evidence_changes_coverage_not_score(patient, trial)
    assert result["coverage_decreased"] is True
    assert result["score_not_forced_to_zero"] is True


def test_esr_evidence_never_reorders_clinical_retrieval() -> None:
    patient = PatientProfile(condition="cancer")
    trial_a = Trial(nct_id="A", title="Cancer trial A", status="RECRUITING", conditions=["Cancer"])
    trial_b = Trial(nct_id="B", title="Cancer trial B", status="RECRUITING", conditions=["Cancer"])
    assert esr_does_not_change_clinical_ordering(patient, trial_a, trial_b)


def test_run_esr_checks_returns_all_three_checks_passing() -> None:
    checks = run_esr_checks()
    assert checks["deterministic"] is True
    assert checks["missing_evidence"]["coverage_decreased"] is True
    assert checks["missing_evidence"]["score_not_forced_to_zero"] is True
    assert checks["does_not_change_clinical_ordering"] is True
