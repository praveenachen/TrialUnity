import SwiftUI

/// Predicted (not observed) under-representation risk. Deliberately styled
/// differently from ESRScoreView -- a thin neutral-slate border (not the ESR
/// module's blue branded surface), a flask symbol, and an "Experimental
/// prediction" label -- so it can never be mistaken for, or read as equally
/// authoritative as, the observed-evidence ESR score above it.
struct RepresentationRiskView: View {
    let risk: RepresentationRiskPrediction

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
            HStack(spacing: Theme.Spacing.xs) {
                Image(systemName: "flask")
                    .font(.caption)
                    .foregroundStyle(Theme.Color.experimental)
                Text("EXPERIMENTAL PREDICTION")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(Theme.Color.experimental)
                    .tracking(0.5)
            }

            HStack {
                Text("Representation risk")
                    .font(.subheadline)
                    .foregroundStyle(Theme.Color.muted)
                Spacer()
                Text(RiskLevelDisplay.label(risk.risk_level))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.Color.ink)
            }
            .accessibilityElement(children: .combine)

            if let confidence = risk.confidence, confidence.isFinite {
                HStack {
                    Text("Confidence")
                        .font(.subheadline)
                        .foregroundStyle(Theme.Color.muted)
                    Spacer()
                    Text(ScoreFormat.clamped(confidence), format: .percent.precision(.fractionLength(0)))
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.Color.ink)
                }
            }

            if !risk.drivers.isEmpty {
                VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                    Text("Drivers")
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

            Divider().overlay(Theme.Color.hairline)

            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                ForEach(risk.limitations, id: \.self) { limitation in
                    Text(limitation)
                        .font(.caption2)
                        .foregroundStyle(Theme.Color.muted)
                }
            }
        }
        .padding(Theme.Spacing.m)
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous)
                .stroke(Theme.Color.hairline, lineWidth: 1)
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
