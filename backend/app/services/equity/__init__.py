"""Equity/Access Representation (ESR) scoring.

Kept import-light: `backend.app.models.schemas` imports `ESRResult` from
`equity.schemas` to embed it in `TrialRecommendation`, so this package must not
eagerly import `equity.service` (which itself imports `models.schemas`) or that
becomes a circular import. Import what you need from the specific submodule:

    from backend.app.services.equity.service import compute_esr, WEIGHTS
    from backend.app.services.equity.race import RaceBenchmark
    from backend.app.services.equity.schemas import ESRResult, ComponentEvidence
"""
