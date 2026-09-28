"""BM25 lexical retrieval over trial searchable text."""
import math

from rank_bm25 import BM25Okapi

from backend.app.models.schemas import Trial
from backend.app.services.retrieval.documents import trial_document
from backend.app.services.text import tokenize_list


class BM25Retriever:
    """Scores trials against a query using BM25Okapi, normalized to [0, 1]."""

    def score(
        self,
        query: str,
        trials: list[Trial],
        *,
        query_tokens: list[str] | None = None,
        document_tokens: list[list[str]] | None = None,
    ) -> list[float]:
        if not trials:
            return []

        query_tokens = query_tokens if query_tokens is not None else tokenize_list(query)
        if not query_tokens:
            return [0.0] * len(trials)

        corpus = document_tokens or [tokenize_list(trial_document(trial)) for trial in trials]
        if not any(corpus):
            return [0.0] * len(trials)

        bm25 = BM25Okapi(corpus)
        # rank-bm25's Okapi IDF can be negative in small candidate sets when a
        # term appears in most documents. Use the standard positive-IDF BM25
        # variant so an exact match remains positive even for one candidate.
        for term in bm25.idf:
            document_frequency = sum(term in frequencies for frequencies in bm25.doc_freqs)
            bm25.idf[term] = math.log(
                1.0 + (len(corpus) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
        raw_scores = bm25.get_scores(query_tokens)
        return _normalize(raw_scores)


def _normalize(scores) -> list[float]:
    """Normalize positive BM25 evidence without making the minimum score zero."""
    positive = [max(0.0, float(score)) for score in scores]
    hi = max(positive, default=0.0)
    if hi < 1e-9:
        return [0.0 for _ in scores]
    return [score / hi for score in positive]
