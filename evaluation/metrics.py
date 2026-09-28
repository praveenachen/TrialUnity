"""Standard IR metrics, kept dependency-free and deterministic.

Recall@K and MRR treat a candidate as "relevant" once its graded relevance is at
or above RELEVANT_THRESHOLD; nDCG@K uses the full graded relevance (0/1/2) and so
is the more informative of the three for this benchmark.
"""
import math

RELEVANT_THRESHOLD = 1


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return round(hits / len(relevant_ids), 6)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for position, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in relevant_ids:
            return round(1.0 / position, 6)
    return 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum((2 ** relevance - 1) / math.log2(index + 2) for index, relevance in enumerate(relevances[:k]))


def ndcg_at_k(ranked_relevances: list[int], k: int) -> float:
    ideal = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    if ideal <= 0:
        return 0.0
    return round(dcg_at_k(ranked_relevances, k) / ideal, 6)
