from backend.app.services.clinicaltrials import ClinicalTrialsClient
from backend.app.services.demographics import BASELINE_RESULTS_SOURCE


def test_normalize_v2_study_payload() -> None:
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT123", "briefTitle": "A test study"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "descriptionModule": {"briefSummary": "This is a trial summary."},
            "conditionsModule": {"conditions": ["Cancer"]},
            "designModule": {"phases": ["PHASE2"]},
            "eligibilityModule": {
                "eligibilityCriteria": "Adults only.",
                "sex": "ALL",
                "minimumAge": "18 Years",
            },
            "contactsLocationsModule": {
                "locations": [{"city": "Toronto", "state": "Ontario", "country": "Canada"}]
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Example Sponsor"}},
            "armsInterventionsModule": {"interventions": [{"name": "Drug A"}]},
        }
    }

    trial = ClinicalTrialsClient()._normalize_study(payload)

    assert trial.nct_id == "NCT123"
    assert trial.conditions == ["Cancer"]
    assert trial.locations == ["Toronto, Ontario, Canada"]
    assert trial.source_url.endswith("/NCT123")


def test_normalize_reported_sex_and_race_prefers_total_group_without_double_counting() -> None:
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCTRESULTS", "briefTitle": "Results study"},
            "eligibilityModule": {"sex": "ALL"},
        },
        "resultsSection": {"baselineCharacteristicsModule": {
            "groups": [
                {"id": "A", "title": "Treatment"},
                {"id": "B", "title": "Control"},
                {"id": "T", "title": "Total", "description": "Total of all reporting groups"},
            ],
            "measures": [
                {"title": "Ethnicity (NIH/OMB)", "paramType": "COUNT_OF_PARTICIPANTS", "classes": [{"categories": [
                    {"title": "Hispanic or Latino", "measurements": [{"groupId": "T", "value": "20"}]},
                ]}]},
                {"title": "Race (NIH/OMB)", "paramType": "COUNT_OF_PARTICIPANTS", "classes": [{"categories": [
                    {"title": "White", "measurements": [
                        {"groupId": "A", "value": "30"}, {"groupId": "B", "value": "20"},
                        {"groupId": "T", "value": "50"},
                    ]},
                    {"title": "Black or African American", "measurements": [
                        {"groupId": "A", "value": "10"}, {"groupId": "B", "value": "15"},
                        {"groupId": "T", "value": "25"},
                    ]},
                    {"title": "Middle Eastern or North African", "measurements": [
                        {"groupId": "T", "value": "5"},
                    ]},
                    {"title": "Unknown or Not Reported", "measurements": [{"groupId": "T", "value": "0"}]},
                ]}]},
                {"title": "Sex: Female, Male", "paramType": "COUNT_OF_PARTICIPANTS", "classes": [{"categories": [
                    {"title": "Female", "measurements": [
                        {"groupId": "A", "value": "40"}, {"groupId": "B", "value": "30"},
                        {"groupId": "T", "value": "70"},
                    ]},
                    {"title": "Male", "measurements": [
                        {"groupId": "A", "value": "20"}, {"groupId": "B", "value": "10"},
                        {"groupId": "T", "value": "30"},
                    ]},
                ]}]},
            ],
        }},
    }

    trial = ClinicalTrialsClient()._normalize_study(payload)

    assert trial.sex == "ALL"
    assert trial.enrollment_sex_distribution == {"FEMALE": 70.0, "MALE": 30.0}
    assert trial.enrollment_race_distribution == {
        "WHITE": 62.5,
        "BLACK": 31.25,
        "MIDDLE_EASTERN_OR_NORTH_AFRICAN": 6.25,
        "UNKNOWN_OR_NOT_REPORTED": 0.0,
    }
    assert trial.enrollment_sex_source == BASELINE_RESULTS_SOURCE
    assert trial.enrollment_race_source == BASELINE_RESULTS_SOURCE


def test_percentage_demographics_are_denominator_weighted_across_arms() -> None:
    payload = {
        "protocolSection": {"identificationModule": {"nctId": "NCTPCT", "briefTitle": "Percent study"}},
        "resultsSection": {"baselineCharacteristicsModule": {
            "groups": [{"id": "A", "title": "A"}, {"id": "B", "title": "B"}],
            "denoms": [{"units": "Participants", "counts": [
                {"groupId": "A", "value": "100"}, {"groupId": "B", "value": "300"},
            ]}],
            "measures": [{"title": "Gender", "paramType": "PERCENTAGE", "unitOfMeasure": "Percent", "classes": [{"categories": [
                {"title": "Female", "measurements": [
                    {"groupId": "A", "value": "50"}, {"groupId": "B", "value": "25"},
                ]},
                {"title": "Male", "measurements": [
                    {"groupId": "A", "value": "50"}, {"groupId": "B", "value": "75"},
                ]},
            ]}]}],
        }},
    }
    trial = ClinicalTrialsClient()._normalize_study(payload)
    assert trial.enrollment_sex_distribution == {"FEMALE": 31.25, "MALE": 68.75}


def test_absent_or_malformed_results_are_ignored() -> None:
    client = ClinicalTrialsClient()
    absent = client._normalize_study({
        "protocolSection": {"identificationModule": {"nctId": "NCTNONE", "briefTitle": "No results"}}
    })
    malformed = client._normalize_study({
        "protocolSection": {"identificationModule": {"nctId": "NCTBAD", "briefTitle": "Bad results"}},
        "resultsSection": {"baselineCharacteristicsModule": {"measures": [
            {"title": "Sex", "classes": [{"categories": [
                {"title": "Female", "measurements": [{"value": "not available"}]},
                {"title": None, "measurements": "wrong shape"},
            ]}]},
            "wrong shape",
        ]}},
    })
    assert absent.enrollment_sex_distribution is None
    assert absent.enrollment_race_distribution is None
    assert malformed.enrollment_sex_distribution is None
    assert malformed.enrollment_race_distribution is None
    assert malformed.enrollment_sex_source is None
