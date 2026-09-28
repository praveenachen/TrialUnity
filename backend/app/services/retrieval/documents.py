"""Searchable text shared by both retrieval backends.

Deliberately excludes ``eligibility_criteria``: that free-text field is long,
often boilerplate, and must never silently inflate a trial's relevance score.
Structured eligibility is judged separately and deterministically (see
``backend.app.services.eligibility``); matching it into the ranking signal
would blur "this trial is relevant" with "this trial says you may qualify".
"""
from backend.app.models.schemas import Trial


def trial_document(trial: Trial) -> str:
    return " ".join(
        part
        for part in [
            trial.title,
            " ".join(trial.conditions),
            " ".join(trial.interventions),
            " ".join(trial.phases),
            trial.brief_summary or "",
            " ".join(trial.locations),
        ]
        if part
    )
