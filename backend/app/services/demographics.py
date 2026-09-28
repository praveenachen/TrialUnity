"""Normalize reported ClinicalTrials.gov baseline demographic distributions."""
import re
from collections.abc import Callable
from typing import Any


BASELINE_RESULTS_SOURCE = "ClinicalTrials.gov resultsSection.baselineCharacteristicsModule"

_SEX_LABELS = {
    "female": "FEMALE",
    "male": "MALE",
    "intersex": "INTERSEX",
    "unknown": "UNKNOWN",
    "unknown or not reported": "UNKNOWN_OR_NOT_REPORTED",
    "not reported": "UNKNOWN_OR_NOT_REPORTED",
}

_RACE_LABELS = {
    "white": "WHITE",
    "black": "BLACK",
    "black or african american": "BLACK",
    "african american": "BLACK",
    "asian": "ASIAN",
    "american indian or alaska native": "AMERICAN_INDIAN_OR_ALASKA_NATIVE",
    "native hawaiian or other pacific islander": "NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER",
    "more than one race": "MULTIPLE_RACES",
    "multiple race": "MULTIPLE_RACES",
    "multiple races": "MULTIPLE_RACES",
    "other": "OTHER",
    "hispanic or latino": "HISPANIC_OR_LATINO",
    "not hispanic or latino": "NOT_HISPANIC_OR_LATINO",
    "unknown": "UNKNOWN",
    "unknown or not reported": "UNKNOWN_OR_NOT_REPORTED",
    "not reported": "UNKNOWN_OR_NOT_REPORTED",
}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).casefold()).strip()


def _preserved_label(value: Any) -> str | None:
    label = re.sub(r"[^A-Z0-9]+", "_", _text(value).upper()).strip("_")
    return label or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    candidate = _text(value).replace(",", "").removesuffix("%")
    if not re.fullmatch(r"\d+(?:\.\d+)?", candidate):
        return None
    return float(candidate)


def _is_total_group(group: dict[str, Any]) -> bool:
    title = _key(group.get("title"))
    description = _key(group.get("description"))
    return title in {"total", "overall", "all participants"} or "total of all reporting groups" in description


def _denominators(module: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for denominator in module.get("denoms", []) or []:
        if not isinstance(denominator, dict):
            continue
        for count in denominator.get("counts", []) or []:
            if not isinstance(count, dict):
                continue
            group_id, value = _text(count.get("groupId")), _number(count.get("value"))
            if group_id and value is not None and value > 0:
                result.setdefault(group_id, value)
    return result


def _category_value(
    measurements: Any,
    *,
    total_groups: set[str],
    denominators: dict[str, float],
    percentage: bool,
) -> float | None:
    values: list[tuple[str, float]] = []
    for measurement in measurements if isinstance(measurements, list) else []:
        if not isinstance(measurement, dict):
            continue
        value = _number(measurement.get("value"))
        if value is not None:
            values.append((_text(measurement.get("groupId")), value))
    if not values:
        return None

    reported_total = [value for group_id, value in values if group_id in total_groups]
    if reported_total:
        return sum(reported_total)

    arm_values = [(group_id, value) for group_id, value in values if group_id not in total_groups]
    if not percentage:
        return sum(value for _, value in arm_values)
    if len(arm_values) == 1:
        return arm_values[0][1]

    weighted = [
        (value, denominators[group_id])
        for group_id, value in arm_values
        if group_id in denominators
    ]
    if len(weighted) != len(arm_values) or not weighted:
        return None
    total_denominator = sum(denominator for _, denominator in weighted)
    return sum(value * denominator for value, denominator in weighted) / total_denominator


def _distribution(
    module: dict[str, Any],
    measure_matches: Callable[[str], bool],
    labels: dict[str, str],
) -> dict[str, float] | None:
    groups = [group for group in module.get("groups", []) or [] if isinstance(group, dict)]
    total_groups = {_text(group.get("id")) for group in groups if _is_total_group(group)}
    denominators = _denominators(module)

    for measure in module.get("measures", []) or []:
        if not isinstance(measure, dict) or not measure_matches(_key(measure.get("title"))):
            continue
        parameter = _key(measure.get("paramType"))
        unit = _key(measure.get("unitOfMeasure"))
        percentage = "percent" in parameter or "percent" in unit
        distribution: dict[str, float] = {}
        for measure_class in measure.get("classes", []) or []:
            if not isinstance(measure_class, dict):
                continue
            for category in measure_class.get("categories", []) or []:
                if not isinstance(category, dict):
                    continue
                raw_label = category.get("title") or measure_class.get("title")
                label = labels.get(_key(raw_label)) or _preserved_label(raw_label)
                value = _category_value(
                    category.get("measurements"),
                    total_groups=total_groups,
                    denominators=denominators,
                    percentage=percentage,
                )
                if label and value is not None:
                    distribution[label] = distribution.get(label, 0.0) + value
        if distribution and sum(distribution.values()) > 0:
            total = sum(distribution.values())
            return {
                label: round((value / total) * 100, 4)
                for label, value in distribution.items()
            }
    return None


def parse_baseline_demographics(
    baseline_module: Any,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Return observed sex and race/ethnicity distributions when usable."""
    if not isinstance(baseline_module, dict):
        return None, None
    sex = _distribution(
        baseline_module,
        lambda title: title == "sex" or title.startswith("sex ") or title == "gender",
        _SEX_LABELS,
    )
    race = _distribution(
        baseline_module,
        lambda title: "race" in title,
        _RACE_LABELS,
    )
    if race is None:
        race = _distribution(
            baseline_module,
            lambda title: "ethnicity" in title,
            _RACE_LABELS,
        )
    return sex, race
