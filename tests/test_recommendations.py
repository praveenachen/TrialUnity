import pytest

from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.recommendations import RecommendationService
from backend.app.services.retrieval import WEIGHTS
from backend.app.services.retrieval.dense import DenseRetriever
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval.lexical import BM25Retriever, _normalize


def test_recommendations_rank_condition_match_highest() -> None:
    service = RecommendationService()
    profile = PatientProfile(condition="lung cancer", location="Toronto", intervention_preferences=["immunotherapy"])
    trials = [
        Trial(
            nct_id="NCT1",
            title="Lung cancer immunotherapy study",
            status="RECRUITING",
            conditions=["Lung Cancer"],
            interventions=["Immunotherapy"],
            brief_summary="A study for lung cancer treatment.",
            locations=["Toronto, Ontario, Canada"],
        ),
        Trial(
            nct_id="NCT2",
            title="Diabetes lifestyle coaching",
            status="RECRUITING",
            conditions=["Type 2 Diabetes"],
            interventions=["Lifestyle Coaching"],
            brief_summary="A study for diabetes self-management.",
            locations=["Remote"],
        ),
    ]

    results = service.recommend(profile, trials)

    assert results[0].trial.nct_id == "NCT1"
    assert results[0].score > results[1].score
    assert "lung" in results[0].explanation.matched_terms


def test_recommendations_include_eligibility_notes() -> None:
    service = RecommendationService()
    profile = PatientProfile(condition="heart failure")
    trial = Trial(
        nct_id="NCT3",
        title="Heart failure monitoring",
        status="RECRUITING",
        conditions=["Heart Failure"],
        sex="ALL",
        minimum_age="21 Years",
        maximum_age="80 Years",
        eligibility_criteria="Adults with heart failure may be eligible.",
    )

    result = service.recommend(profile, [trial])[0]

    assert result.explanation.eligibility_notes
    assert result.explanation.patient_friendly_summary


def test_score_breakdown_is_returned_and_sums_to_score() -> None:
    service = RecommendationService()
    profile = PatientProfile(condition="lung cancer", location="Toronto", intervention_preferences=["immunotherapy"])
    trial = Trial(
        nct_id="NCT1", title="Lung cancer immunotherapy study", status="RECRUITING",
        conditions=["Lung Cancer"], interventions=["Immunotherapy"],
        brief_summary="A study for lung cancer treatment.", locations=["Toronto, Ontario, Canada"],
    )

    result = service.recommend(profile, [trial])[0]

    assert set(result.score_breakdown) == set(WEIGHTS)
    assert round(sum(result.score_breakdown.values()), 4) == result.score
    # Structured signals matched, so their full weight should be credited.
    assert result.score_breakdown["condition"] == WEIGHTS["condition"]
    assert result.score_breakdown["intervention"] == WEIGHTS["intervention"]
    assert result.score_breakdown["location"] == WEIGHTS["location"]


def test_matched_terms_do_not_add_a_score_bonus_beyond_lexical_weight() -> None:
    # A trial that only overlaps on stray terms (no structured match) should score
    # exactly its lexical + semantic contribution -- overlap terms are explanatory only.
    retriever = HybridRetriever()
    profile = PatientProfile(condition="lung cancer")
    trial = Trial(nct_id="1", title="Lung cancer biology overview", conditions=["Oncology"])

    (_, score, breakdown, _components, matched_terms) = retriever.rank(profile, [trial])[0]

    assert matched_terms  # "lung" and/or "cancer" overlap
    assert breakdown["condition"] == 0  # "lung cancer" not in trial.conditions verbatim
    assert round(breakdown["lexical"] + breakdown["semantic"], 3) == round(score, 3)


def test_structured_fields_and_eligibility_text_do_not_change_relevance() -> None:
    service = RecommendationService()
    trial = Trial(nct_id="1", title="Cancer", conditions=["Cancer"])
    changed = trial.model_copy(update={"sex": "FEMALE", "minimum_age": "90 Years",
                                      "eligibility_criteria": "Cancer cancer cancer"})
    first = service.recommend(PatientProfile(condition="cancer", sex="female", age=40), [trial])[0]
    second = service.recommend(PatientProfile(condition="cancer", sex="male", age=50), [changed])[0]
    assert first.score == second.score


def test_empty_and_single_trial_candidate_sets_do_not_fail() -> None:
    service = RecommendationService()
    patient = PatientProfile(condition="cancer")
    assert service.recommend(patient, []) == []

    result = service.recommend(patient, [Trial(nct_id="1", title="Unrelated diabetes coaching study")])[0]
    assert 0 <= result.score <= 1


def test_lexical_relevance_contributes_correctly() -> None:
    # Three candidates so a term appearing in only one document has a non-zero IDF
    # (BM25's classic idf collapses to 0 for a term split exactly 1-of-2 documents).
    scores = BM25Retriever().score("lung cancer immunotherapy", [
        Trial(nct_id="1", title="Lung cancer immunotherapy trial"),
        Trial(nct_id="2", title="Diabetes lifestyle coaching program"),
        Trial(nct_id="3", title="Seasonal allergy nasal spray study"),
    ])
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert BM25Retriever().score("lung cancer", []) == []
    assert BM25Retriever().score("the", [Trial(nct_id="1", title="the")]) == [0.0]
    assert BM25Retriever().score("lung cancer", [
        Trial(nct_id="1", title="Lung cancer")
    ]) == [1.0]


def test_bm25_normalization_preserves_equal_positive_evidence() -> None:
    assert _normalize([2.0, 2.0]) == [1.0, 1.0]
    assert _normalize([-1.0, 0.0, 2.0]) == [0.0, 0.0, 1.0]


def test_semantic_cosine_is_clipped_without_half_score_baseline(monkeypatch) -> None:
    import numpy as np
    from backend.app.services.retrieval import dense

    class FakeModel:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            return np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    monkeypatch.setattr(dense, "_get_model", lambda: FakeModel())
    scores = DenseRetriever().score("query", [
        Trial(nct_id="1", title="same"),
        Trial(nct_id="2", title="orthogonal"),
        Trial(nct_id="3", title="opposite"),
    ])
    assert scores == [1.0, 0.0, 0.0]
    assert DenseRetriever().score("lung cancer", []) == []


def test_hybrid_ranking_beats_obviously_irrelevant_candidates() -> None:
    retriever = HybridRetriever()
    profile = PatientProfile(condition="lung cancer", intervention_preferences=["immunotherapy"])
    relevant = Trial(
        nct_id="1", title="Lung cancer immunotherapy trial", conditions=["Lung Cancer"],
        interventions=["Immunotherapy"], brief_summary="A study of immunotherapy for lung cancer patients.",
    )
    irrelevant = Trial(
        nct_id="2", title="Seasonal allergy nasal spray study", conditions=["Allergic Rhinitis"],
        interventions=["Nasal Spray"], brief_summary="A study of a nasal spray for seasonal allergies.",
    )

    ranked = retriever.rank(profile, [irrelevant, relevant])

    assert [trial.nct_id for trial, *_ in ranked] == ["1", "2"]


def test_eligibility_is_independent_of_relevance_score() -> None:
    # Two patients with identical retrieval-relevant text but different structured
    # eligibility outcomes must still get the same relevance score.
    service = RecommendationService()
    trial = Trial(nct_id="1", title="Cancer trial", conditions=["Cancer"], status="RECRUITING",
                  minimum_age="18 Years", maximum_age="99 Years", sex="ALL")

    eligible = service.recommend(PatientProfile(condition="cancer", age=40, sex="female"), [trial])[0]
    ineligible = service.recommend(PatientProfile(condition="cancer", age=10, sex="female"), [trial])[0]

    assert eligible.score == ineligible.score
    assert eligible.structured_eligibility.status == "compatible"
    assert ineligible.structured_eligibility.status == "incompatible"


class StaticRetriever:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, trials, **kwargs):
        return [self.scores[trial.nct_id] for trial in trials]


def configured_retriever(lexical, semantic) -> HybridRetriever:
    retriever = HybridRetriever()
    retriever._lexical = StaticRetriever(lexical)
    retriever._dense = StaticRetriever(semantic)
    return retriever


def test_exact_condition_beats_semantically_related_condition() -> None:
    trials = [
        Trial(nct_id="exact", title="NSCLC", conditions=["Lung Cancer"]),
        Trial(nct_id="related", title="Thoracic malignancy", conditions=["Thoracic Neoplasm"]),
    ]
    ranked = configured_retriever(
        {"exact": 0.7, "related": 0.1}, {"exact": 0.8, "related": 0.9}
    ).rank(PatientProfile(condition="lung cancer"), trials)
    assert [item[0].nct_id for item in ranked] == ["exact", "related"]


def test_semantic_match_can_win_when_lexical_overlap_is_weak() -> None:
    trials = [Trial(nct_id="semantic", title="A"), Trial(nct_id="other", title="B")]
    ranked = configured_retriever(
        {"semantic": 0.0, "other": 0.1}, {"semantic": 0.9, "other": 0.1}
    ).rank(PatientProfile(condition="myocardial infarction"), trials)
    assert ranked[0][0].nct_id == "semantic"


def test_strong_lexical_match_can_win_when_semantics_are_weaker() -> None:
    trials = [Trial(nct_id="lexical", title="A"), Trial(nct_id="semantic", title="B")]
    ranked = configured_retriever(
        {"lexical": 1.0, "semantic": 0.1}, {"lexical": 0.1, "semantic": 0.4}
    ).rank(PatientProfile(condition="EGFR exon 20"), trials)
    assert ranked[0][0].nct_id == "lexical"


def test_incidental_token_overlap_cannot_beat_clearly_relevant_trial() -> None:
    trials = [
        Trial(nct_id="relevant", title="A", conditions=["Lung Cancer"]),
        Trial(nct_id="incidental", title="B", conditions=["Diabetes"]),
    ]
    ranked = configured_retriever(
        {"relevant": 0.3, "incidental": 1.0}, {"relevant": 0.9, "incidental": 0.0}
    ).rank(PatientProfile(condition="lung cancer"), trials)
    assert ranked[0][0].nct_id == "relevant"


@pytest.mark.parametrize(
    ("profile_changes", "trial_changes", "component"),
    [
        ({"location": "Toronto"}, {"locations": ["Toronto, Canada"]}, "location"),
        ({"phase_preferences": ["PHASE2"]}, {"phases": ["phase2"]}, "phase"),
        ({"intervention_preferences": ["Drug A"]}, {"interventions": ["Drug A"]}, "intervention"),
    ],
)
def test_structured_preferences_influence_rank_once(profile_changes, trial_changes, component) -> None:
    trials = [
        Trial(nct_id="match", title="A", **trial_changes),
        Trial(nct_id="other", title="B"),
    ]
    ranked = configured_retriever(
        {"match": 0.0, "other": 0.0}, {"match": 0.0, "other": 0.0}
    ).rank(PatientProfile(condition="cancer", **profile_changes), trials)
    assert ranked[0][0].nct_id == "match"
    assert ranked[0][3][component] == 1.0
    assert ranked[1][3][component] == 0.0


def test_preferences_are_not_duplicated_in_lexical_or_semantic_query() -> None:
    captured_queries = []

    class CapturingRetriever:
        def score(self, query, trials, **kwargs):
            captured_queries.append(query)
            return [0.0] * len(trials)

    retriever = HybridRetriever()
    retriever._lexical = CapturingRetriever()
    retriever._dense = CapturingRetriever()
    retriever.rank(
        PatientProfile(
            condition="cancer",
            notes="advanced disease",
            location="Toronto",
            phase_preferences=["PHASE2"],
            intervention_preferences=["Drug A"],
        ),
        [Trial(nct_id="1", title="Cancer")],
    )
    assert captured_queries == ["cancer advanced disease", "cancer advanced disease"]


def test_notes_still_influence_downstream_ranking() -> None:
    profile = PatientProfile(condition="cancer", notes="metastatic biomarker recurrence")
    trials = [
        Trial(nct_id="notes-match", title="Cancer study", conditions=["Cancer"],
              brief_summary="Metastatic biomarker recurrence cohort."),
        Trial(nct_id="other-1", title="Cancer study", conditions=["Cancer"],
              brief_summary="General supportive care."),
        Trial(nct_id="other-2", title="Cancer study", conditions=["Cancer"],
              brief_summary="Routine symptom monitoring."),
    ]
    results = RecommendationService().recommend(profile, trials)
    assert results[0].trial.nct_id == "notes-match"
    assert "biomarker" in results[0].explanation.matched_terms


def test_equivalent_scores_use_nct_id_as_deterministic_tie_breaker() -> None:
    trials = [Trial(nct_id="NCT2", title="Same"), Trial(nct_id="NCT1", title="Same")]
    ranked = configured_retriever(
        {"NCT1": 0.5, "NCT2": 0.5}, {"NCT1": 0.5, "NCT2": 0.5}
    ).rank(PatientProfile(condition="cancer"), trials)
    assert [item[0].nct_id for item in ranked] == ["NCT1", "NCT2"]


def test_weights_sum_to_one_and_api_exposes_normalized_and_weighted_scores() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    result = RecommendationService().recommend(
        PatientProfile(condition="cancer"),
        [Trial(nct_id="1", title="Cancer", conditions=["Cancer"])],
    )[0]
    assert result.relevance.overall == result.score
    assert result.relevance.weighted_components == result.score_breakdown
    assert result.relevance.weights == WEIGHTS
    assert result.relevance.structured["condition"] == 1.0
    assert result.relevance.lexical >= 0
    assert result.relevance.semantic >= 0


def test_trial_documents_are_built_once_per_ranking_request(monkeypatch) -> None:
    from backend.app.services.retrieval import hybrid

    calls = []
    original = hybrid.trial_document

    def counted(trial):
        calls.append(trial.nct_id)
        return original(trial)

    monkeypatch.setattr(hybrid, "trial_document", counted)
    retriever = configured_retriever({"1": 0.0, "2": 0.0}, {"1": 0.0, "2": 0.0})
    retriever.rank(PatientProfile(condition="cancer"), [
        Trial(nct_id="1", title="A"), Trial(nct_id="2", title="B")
    ])
    assert calls == ["1", "2"]


@pytest.mark.model_cache
def test_sentence_transformer_is_cached_once_per_process(monkeypatch) -> None:
    from backend.app.services.retrieval import dense

    loads = []

    class Model:
        pass

    def load(name):
        loads.append(name)
        return Model()

    dense._get_model.cache_clear()
    monkeypatch.setattr(dense, "SentenceTransformer", load)
    assert dense._get_model() is dense._get_model()
    assert loads == ["sentence-transformers/all-MiniLM-L6-v2"]
    dense._get_model.cache_clear()
