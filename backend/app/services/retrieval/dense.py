"""Dense semantic retrieval using a locally cached SentenceTransformer model."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from backend.app.models.schemas import Trial
from backend.app.services.retrieval.documents import trial_document

# Small (~80MB), CPU-friendly, no API key required.
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Loads the embedding model once per process and reuses it for every request."""
    return SentenceTransformer(_MODEL_NAME)


class DenseRetriever:
    """Scores trials by cosine similarity between a patient query and trial text."""

    def score(
        self,
        query: str,
        trials: list[Trial],
        *,
        documents: list[str] | None = None,
    ) -> list[float]:
        if not trials:
            return []

        model = _get_model()
        documents = documents or [trial_document(trial) for trial in trials]
        embeddings = model.encode([query, *documents], normalize_embeddings=True)
        query_vector, trial_vectors = embeddings[0], embeddings[1:]

        # Negative/orthogonal cosine values provide no positive semantic evidence.
        similarities = trial_vectors @ query_vector
        return [round(max(0.0, min(1.0, float(similarity))), 4) for similarity in similarities]
