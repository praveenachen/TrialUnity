import SwiftUI

/// Predicted (not observed) under-representation risk. Deliberately styled
/// differently from ESRScoreView -- a dashed border, a flask symbol, and an
/// "EXPERIMENTAL / PREDICTED" label -- so it can never be mistaken for the
/// authoritative, evidence-based ESR score above it.
struct RepresentationRiskView: View {
    let risk: RepresentationRiskPrediction

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
            HStack(spacing: Theme.Spacing.xs) {
                Image(systemName: "flask")
                    .foregroundStyle(Theme.Color.muted)
                Text("EXPERIMENTAL · PREDICTED · NOT OBSERVED EVIDENCE")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(Theme.Color.muted)
            }

            HStack(alignment: .firstTextBaseline, spacing: Theme.Spacing.s) {
                Text("Representation risk")
                    .font(.sectionLabel)
                    .foregroundStyle(Theme.Color.muted)
                Text(RiskLevelDisplay.label(risk.risk_level))
                    .font(.editorialTitle)
                    .foregroundStyle(Theme.Color.ink)
            }
            .accessibilityElement(children: .combine)

            if let confidence = risk.confidence, confidence.isFinite {
                Text("Confidence: \(ScoreFormat.clamped(confidence), format: .percent.precision(.fractionLength(0)))")
                    .font(.subheadline)
                    .foregroundStyle(Theme.Color.muted)
            }

            if !risk.drivers.isEmpty {
                VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                    Text("Top drivers")
                        .font(.sectionLabel)
                        .foregroundStyle(Theme.Color.muted)
                    ForEach(risk.drivers, id: \.self) { driver in
                        Label(RiskLevelDisplay.driverLabel(driver), systemImage: "circle.fill")
                            .font(.caption)
                            .foregroundStyle(Theme.Color.ink)
                            .imageScale(.small)
                    }
                }
            }

            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                ForEach(risk.limitations, id: \.self) { limitation in
                    Text(limitation)
                        .font(.caption)
                        .foregroundStyle(Theme.Color.muted)
                }
            }
        }
        .padding(Theme.Spacing.m)
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous)
                .strokeBorder(Theme.Color.hairline, style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
        )
        .accessibilityElement(children: .contain)
    }
}

#Preview {
    RepresentationRiskView(risk: RepresentationRiskPrediction(
        risk_level: "low",
        probabilities: ["low": 0.59, "moderate": 0.3, "high": 0.11],
        confidence: 0.59,
        model_version: "representation-risk-experimental-v1",
        evidence_type: "predicted",
        drivers: ["num_sites (importance 0.552)", "num_regions (importance 0.465)", "target_enrollment_log (importance 0.301)"],
        limitations: [
            "EXPERIMENTAL and NOT VALIDATED: trained and evaluated on synthetic fixture data only, not real historical trials.",
            "Never a substitute for observed ESR evidence when real enrollment demographics are available.",
        ]
    ))
    .padding()
}
