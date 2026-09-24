"""Conservative checks of structured fields only; never full medical eligibility."""
import re

from backend.app.models.schemas import (
    EligibilityCriterion, PatientProfile, StructuredEligibility, Trial,
)


def _age_bound(age: int | None, bound: str | None, minimum: bool) -> EligibilityCriterion:
    label = "minimum" if minimum else "maximum"
    if age is None:
        return EligibilityCriterion(state="unknown", reason="Patient age is missing.")
    if not bound:
        return EligibilityCriterion(state="unknown", reason=f"Trial {label} age is missing.")
    if bound.strip().upper() == "N/A":
        return EligibilityCriterion(state="compatible", reason=f"Trial explicitly specifies no {label} age limit.")
    # Intake provides integer years. Do not approximate month/day/week boundaries.
    match = re.fullmatch(r"(\d+)\s+Years?", bound.strip(), re.IGNORECASE)
    if not match:
        return EligibilityCriterion(state="unknown", reason=f"Trial {label} age requires review: {bound}.")
    limit = int(match.group(1))
    incompatible = age < limit if minimum else age > limit
    return EligibilityCriterion(
        state="incompatible" if incompatible else "compatible",
        reason=f"Patient age {age} years; trial {label} age {limit} years.",
    )


def evaluate_eligibility(patient: PatientProfile, trial: Trial) -> StructuredEligibility:
    criteria = {
        "minimum_age": _age_bound(patient.age, trial.minimum_age, True),
        "maximum_age": _age_bound(patient.age, trial.maximum_age, False),
    }
    # Contradictory registry bounds cannot safely establish compatibility.
    bounds = [re.fullmatch(r"(\d+)\s+Years?", (value or "").strip(), re.IGNORECASE)
              for value in (trial.minimum_age, trial.maximum_age)]
    if all(bounds) and int(bounds[0].group(1)) > int(bounds[1].group(1)):
        for key in ("minimum_age", "maximum_age"):
            criteria[key] = EligibilityCriterion(state="unknown", reason="Trial age bounds are contradictory.")

    patient_sex = (patient.sex or "").strip().upper()
    trial_sex = (trial.sex or "").strip().upper()
    sex_state = "unknown"
    if patient_sex in {"MALE", "FEMALE"} and trial_sex in {"MALE", "FEMALE", "ALL"}:
        sex_state = "compatible" if trial_sex in {patient_sex, "ALL"} else "incompatible"
    criteria["sex"] = EligibilityCriterion(
        state=sex_state,
        reason=f"Patient sex: {patient_sex or 'missing'}; trial sex: {trial_sex or 'missing'}.",
    )
    status = trial.status.strip().upper().replace(" ", "_")
    recruitment_state = "unknown"
    if status == "RECRUITING":
        recruitment_state = "compatible"
    elif status in {"ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED", "NO_LONGER_AVAILABLE"}:
        recruitment_state = "incompatible"
    criteria["recruitment_status"] = EligibilityCriterion(
        state=recruitment_state,
        reason=f"Registry recruitment status: {trial.status}; site availability must be confirmed.",
    )
    states = {criterion.state for criterion in criteria.values()}
    overall = "incompatible" if "incompatible" in states else "unknown" if "unknown" in states else "compatible"
    return StructuredEligibility(status=overall, criteria=criteria)
