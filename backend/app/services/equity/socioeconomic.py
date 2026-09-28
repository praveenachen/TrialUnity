"""Socioeconomic Access v1: structural site/geographic access signals only.

This deliberately does NOT model income, transportation, ability to pay, or any
other socioeconomic attribute of the patient -- none of that data exists in
TrialUnity's sources. It scores how physically/logistically reachable a trial's
sites are, using only what ClinicalTrials.gov and the patient intake form already
provide: how many sites a trial has, how geographically spread they are, whether
decentralized/remote participation is offered, and whether a site is near the
patient's stated location.

Extensible for Phase 5: swap in an external area deprivation index or real
travel-time estimates by adding a signal here without changing the component
contract (ComponentEvidence) or the ESR service that calls this module.
"""
from backend.app.models.schemas import PatientProfile, Trial
from backend.app.services.equity.schemas import ComponentEvidence

# Diminishing-returns caps: score reaches 100 once a trial has this many sites /
# distinct regions. Small, easy-to-explain constants -- not fit to any dataset.
SITES_FOR_FULL_REACH_CREDIT = 5
REGIONS_FOR_FULL_SPREAD_CREDIT = 4
DECENTRALIZED_KEYWORDS = ("remote", "virtual", "decentralized", "telehealth", "online")

# Total possible sub-signals; used to report how complete this component's evidence is.
_MAX_SIGNALS = 4


def _region_signature(location: str) -> str:
    """Best-effort state/country grouping from a 'City, State, Country' label."""
    parts = [part.strip() for part in location.split(",") if part.strip()]
    return ", ".join(parts[1:]) if len(parts) > 1 else location.strip()


def socioeconomic_component(patient: PatientProfile, trial: Trial) -> ComponentEvidence:
    locations = [location for location in trial.locations if location and location.strip()]
    signals: dict[str, float] = {}
    missing: list[str] = []

    if locations:
        signals["site_reach"] = min(1.0, len(locations) / SITES_FOR_FULL_REACH_CREDIT) * 100
        regions = {_region_signature(location).casefold() for location in locations}
        signals["geographic_spread"] = min(1.0, len(regions) / REGIONS_FOR_FULL_SPREAD_CREDIT) * 100
        has_decentralized = any(
            keyword in location.casefold() for location in locations for keyword in DECENTRALIZED_KEYWORDS
        )
        signals["decentralized_access"] = 100.0 if has_decentralized else 0.0
    else:
        missing.append("Trial does not list any site locations.")

    if patient.location:
        matched = any(patient.location.casefold() in location.casefold() for location in locations)
        signals["patient_proximity"] = 100.0 if matched else 0.0
    else:
        missing.append("Patient did not provide a location to check proximity against trial sites.")

    if not signals:
        return ComponentEvidence(
            score=None,
            evidence_coverage=0.0,
            evidence_type="insufficient_data",
            rationale="No site location or patient location data is available to assess geographic/site access.",
            source="trial.locations / patient.location",
            missing_evidence=missing,
        )

    score = round(sum(signals.values()) / len(signals), 2)
    coverage = round(len(signals) / _MAX_SIGNALS, 4)
    detail = "; ".join(f"{name.replace('_', ' ')} {value:.0f}/100" for name, value in signals.items())
    return ComponentEvidence(
        score=score,
        evidence_coverage=coverage,
        evidence_type="observed_geographic",
        rationale=(
            "Structural site/geographic access only (site count, regional spread, decentralized "
            f"participation, patient-to-site text proximity) -- not income, transportation, or "
            f"ability to pay, which TrialUnity has no source for. Signals: {detail}."
        ),
        source="ClinicalTrials.gov site locations; patient-reported location",
        missing_evidence=missing,
    )
