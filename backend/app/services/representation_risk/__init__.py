"""Backend-facing serving layer for the EXPERIMENTAL representation-risk model.

Kept import-light for the same reason as `backend.app.services.equity`:
`backend.app.models.schemas` imports `RepresentationRiskPrediction` from
`representation_risk.schemas` to embed it in `TrialRecommendation`, and this
package must not eagerly import `predictor` (which depends on `models.schemas`)
or that becomes a circular import. Import what you need directly:

    from backend.app.services.representation_risk.predictor import predict_representation_risk
    from backend.app.services.representation_risk.schemas import RepresentationRiskPrediction
"""
