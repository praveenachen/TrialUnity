from backend.app.services.clinicaltrials import ClinicalTrialsClient


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
