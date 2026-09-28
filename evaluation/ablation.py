"""Ablation comparison: lexical-only (BM25) vs. dense-only (embeddings) vs. hybrid.

Each system ranks the exact same candidate trials for the exact same query text, so
the only thing that varies is which signals contribute to the ranking. Lexical-only
and dense-only intentionally ignore structured preferences (location/phase/
intervention) because, in the production system, those are additive signals that
only HybridRetriever applies -- they were never part of the text query to begin
with. This is not a benchmark artifact; it's how the app actually works.
"""
from dataclasses import asdict, dataclass

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.retrieval.dense import DenseRetriever
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval.lexical import BM25Retriever

from evaluation.benchmark import BenchmarkCase, load_benchmark
from evaluation.metrics import RELEVANT_THRESHOLD, ndcg_at_k, recall_at_k, reciprocal_rank

DEFAULT_KS = (3, 5)
SYSTEMS = ("lexical", "dense", "hybrid")


def _query_text(patient: PatientProfile) -> str:
    # Mirrors HybridRetriever._profile_text exactly (condition + notes only) so the
    # ablation compares the same query text hybrid retrieval itself uses.
    return " ".join(part for part in [patient.condition, patient.notes or ""] if part)


def _ranked_ids_by_score(trials: list[Trial], scores: list[float]) -> list[str]:
    # Same deterministic tie-break HybridRetriever uses: score desc, then nct_id.
    paired = list(zip((trial.nct_id for trial in trials), scores))
    return [nct_id for nct_id, _score in sorted(paired, key=lambda item: (-item[1], item[0]))]


def _lexical_only_ranking(patient: PatientProfile, trials: list[Trial]) -> list[str]:
    scores = BM25Retriever().score(_query_text(patient), trials)
    return _ranked_ids_by_score(trials, scores)


def _dense_only_ranking(patient: PatientProfile, trials: list[Trial]) -> list[str]:
    scores = DenseRetriever().score(_query_text(patient), trials)
    return _ranked_ids_by_score(trials, scores)


def _hybrid_ranking(patient: PatientProfile, trials: list[Trial]) -> tuple[list[str], list[dict]]:
    ranked = HybridRetriever().rank(patient, trials)
    ids = [trial.nct_id for trial, *_ in ranked]
    rows = []
    for trial, score, breakdown, components, matched_terms in ranked:
        structured_contribution = round(
            sum(breakdown[name] for name in ("condition", "intervention", "phase", "location")), 4
        )
        rows.append({
            "nct_id": trial.nct_id,
            "score": score,
            "lexical_score": components["lexical"],
            "semantic_score": components["semantic"],
            "structured_contribution": structured_contribution,
            "matched_terms": matched_terms,
        })
    return ids, rows


@dataclass
class CaseDiagnostics:
    """Compact enough to eyeball a ranking failure without re-running anything."""

    case_id: str
    category: str
    expected_relevant: list[str]
    hybrid_ranking: list[dict]


def _metrics_for_ranking(ranked_ids: list[str], relevance_by_id: dict[str, int], ks: tuple[int, ...]) -> dict:
    relevant_ids = {nct_id for nct_id, grade in relevance_by_id.items() if grade >= RELEVANT_THRESHOLD}
    ranked_relevances = [relevance_by_id[nct_id] for nct_id in ranked_ids]
    metrics = {"mrr": reciprocal_rank(ranked_ids, relevant_ids)}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_relevances, k)
    return metrics


def evaluate_case(case: BenchmarkCase, ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    trials = case.trials
    relevance_by_id = case.relevance_by_id
    relevant_ids = {nct_id for nct_id, grade in relevance_by_id.items() if grade >= RELEVANT_THRESHOLD}

    hybrid_ids, hybrid_rows = _hybrid_ranking(case.patient, trials)
    rankings = {
        "lexical": _lexical_only_ranking(case.patient, trials),
        "dense": _dense_only_ranking(case.patient, trials),
        "hybrid": hybrid_ids,
    }

    diagnostics = CaseDiagnostics(
        case_id=case.id,
        category=case.category,
        expected_relevant=sorted(relevant_ids),
        hybrid_ranking=[
            {**row, "relevance": relevance_by_id[row["nct_id"]]} for row in hybrid_rows
        ],
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "metrics": {system: _metrics_for_ranking(ranked, relevance_by_id, ks) for system, ranked in rankings.items()},
        "diagnostics": asdict(diagnostics),
    }


def run_ablation(ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    cases = load_benchmark()
    per_case = [evaluate_case(case, ks) for case in cases]

    aggregate = {}
    for system in SYSTEMS:
        metric_names = per_case[0]["metrics"][system].keys()
        aggregate[system] = {
            name: round(sum(case["metrics"][system][name] for case in per_case) / len(per_case), 4)
            for name in metric_names
        }

    return {"num_cases": len(per_case), "ks": list(ks), "aggregate": aggregate, "cases": per_case}
