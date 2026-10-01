from backend.app.services.clinicaltrials import ClinicalTrialsClient


def study(locations):
    return {"protocolSection": {"identificationModule": {"nctId": "NCT123", "briefTitle": "Test"},
                               "contactsLocationsModule": {"locations": locations}}}


def test_site_metadata_preserves_existing_locations():
    trial = ClinicalTrialsClient()._normalize_study(study([
        {"facility": "Toronto Hospital", "city": "Toronto", "state": "Ontario", "country": "Canada",
         "status": "RECRUITING", "geoPoint": {"lat": 43.65, "lon": -79.38}},
        {"city": "Ottawa", "state": "Ontario", "country": "Canada"},
    ]))
    assert trial.locations == ["Toronto, Ontario, Canada", "Ottawa, Ontario, Canada"]
    assert len(trial.trial_sites) == 2
    assert trial.trial_sites[0].facility == "Toronto Hospital"
    assert trial.trial_sites[0].latitude == 43.65
    assert trial.trial_sites[0].status == "RECRUITING"
    assert trial.trial_sites[1].latitude is None
    assert trial.trial_sites[1].status is None


def test_invalid_coordinates_are_not_invented():
    for invalid in ("unavailable", float("nan"), 999):
        trial = ClinicalTrialsClient()._normalize_study(study([
            {"city": "Toronto", "geoPoint": {"lat": invalid, "lon": invalid}}
        ]))
        assert trial.trial_sites[0].latitude is None
        assert trial.trial_sites[0].longitude is None


def test_missing_sites_remain_empty():
    trial = ClinicalTrialsClient()._normalize_study(study([]))
    assert trial.locations == []
    assert trial.trial_sites == []
