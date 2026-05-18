from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.recommendations import RecommendationService


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
