import logging

from pathlib import Path
from typing import Any

import httpx

from backend.app.core.config import get_settings
from backend.app.models.schemas import Trial, TrialSearchRequest
from backend.app.services.demographics import BASELINE_RESULTS_SOURCE, parse_baseline_demographics
from backend.app.services.text import normalize_space


logger = logging.getLogger(__name__)

SAMPLE_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_trials.json"


class ClinicalTrialsClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def search(self, request: TrialSearchRequest) -> tuple[list[Trial], str]:
        params: dict[str, Any] = {
            "format": "json",
            "pageSize": request.page_size,
            "query.term": request.query,
        }
        if request.condition:
            params["query.cond"] = request.condition
        if request.location:
            params["query.locn"] = request.location
        if request.phase:
            params["filter.advanced"] = f"AREA[Phase]{request.phase}"
        if request.recruitment_status:
            params["filter.overallStatus"] = request.recruitment_status

        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(f"{self.settings.ctgov_base_url}/studies", params=params)
                response.raise_for_status()
                payload = response.json()
            studies = payload.get("studies", [])
            trials = [self._normalize_study(study) for study in studies]
            return trials, "clinicaltrials.gov"
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("ClinicalTrials.gov search failed (%s); using sample-data", type(exc).__name__)
            trials = self.sample_trials()
            if request.phase:
                trials = [trial for trial in trials if request.phase in trial.phases]
            if request.recruitment_status:
                statuses = {status.strip().upper() for status in request.recruitment_status.split(",")}
                trials = [trial for trial in trials if trial.status.upper() in statuses]
            return trials[:request.page_size], "sample-data"

    async def get_trial(self, nct_id: str) -> tuple[Trial | None, str]:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(f"{self.settings.ctgov_base_url}/studies/{nct_id}", params={"format": "json"})
                response.raise_for_status()
            return self._normalize_study(response.json()), "clinicaltrials.gov"
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("ClinicalTrials.gov detail failed (%s); using sample-data", type(exc).__name__)
            for trial in self.sample_trials():
                if trial.nct_id.lower() == nct_id.lower():
                    return trial, "sample-data"
            return None, "sample-data"

    def sample_trials(self) -> list[Trial]:
        import json

        payload = json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))
        return [Trial(**item) for item in payload]

    def _normalize_study(self, study: dict[str, Any]) -> Trial:
        protocol = study.get("protocolSection", study.get("Study", {}).get("ProtocolSection", {}))
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        description = protocol.get("descriptionModule", {})
        conditions = protocol.get("conditionsModule", {})
        design = protocol.get("designModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        contacts = protocol.get("contactsLocationsModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {})
        arms = protocol.get("armsInterventionsModule", {})
        results = study.get("resultsSection", {})
        baseline = results.get("baselineCharacteristicsModule", {}) if isinstance(results, dict) else {}
        sex_distribution, race_distribution = parse_baseline_demographics(baseline)

        nct_id = identification.get("nctId") or identification.get("NCTId") or "UNKNOWN"
        locations = []
        trial_sites = []
        for location in contacts.get("locations", []) or []:
            parts = [location.get("city"), location.get("state"), location.get("country")]
            label = ", ".join(part for part in parts if part)
            if label:
                locations.append(label)
                point = location.get("geoPoint") or {}
                def coordinate(value, limit):
                    try:
                        number = float(value)
                        return number if -limit <= number <= limit else None
                    except (TypeError, ValueError):
                        return None
                trial_sites.append({
                    "facility": location.get("facility"), "location": label,
                    "status": location.get("status"),
                    "latitude": coordinate(point.get("lat"), 90),
                    "longitude": coordinate(point.get("lon"), 180),
                })

        interventions = [
            normalize_space(item.get("name"))
            for item in arms.get("interventions", []) or []
            if item.get("name")
        ]

        return Trial(
            nct_id=nct_id,
            title=identification.get("briefTitle") or identification.get("officialTitle") or "Untitled clinical trial",
            status=status.get("overallStatus", "Unknown"),
            conditions=conditions.get("conditions", []) or [],
            interventions=interventions,
            phases=design.get("phases", []) or [],
            brief_summary=description.get("briefSummary"),
            eligibility_criteria=eligibility.get("eligibilityCriteria"),
            sex=eligibility.get("sex"),
            minimum_age=eligibility.get("minimumAge"),
            maximum_age=eligibility.get("maximumAge"),
            locations=locations,
            trial_sites=trial_sites,
            sponsor=(sponsor.get("leadSponsor") or {}).get("name"),
            source_url=f"https://clinicaltrials.gov/study/{nct_id}",
            enrollment_sex_distribution=sex_distribution,
            enrollment_race_distribution=race_distribution,
            enrollment_sex_source=BASELINE_RESULTS_SOURCE if sex_distribution else None,
            enrollment_race_source=BASELINE_RESULTS_SOURCE if race_distribution else None,
        )
