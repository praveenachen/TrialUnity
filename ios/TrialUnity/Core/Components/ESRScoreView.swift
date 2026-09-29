import SwiftUI

/// Representation & Access evidence (ESR). Renders exactly what the backend
/// computed -- a null component score is always shown as "—", never as 0, and
/// evidence coverage is a visible bar, not just a number buried in text.
struct ESRScoreView: View {
    let esr: ESRResult?

    var body: some View {
        if let esr {
            VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                HStack(alignment: .firstTextBaseline) {
                    Text("ESR")
                        .font(.sectionLabel)
                        .foregroundStyle(Theme.Color.muted)
                    Text(ScoreFormat.rounded(esr.score))
                        .font(.editorialTitle)
                        .foregroundStyle(Theme.Color.ink)
                    if esr.score == nil {
                        Text("Not enough evidence")
                            .font(.caption)
                            .foregroundStyle(Theme.Color.muted)
                    }
                }
                .accessibilityElement(children: .combine)

                CoverageBar(coverage: esr.evidence_coverage)

                VStack(spacing: 0) {
                    ForEach(Array(ESRDisplay.orderedComponents(esr.components).enumerated()), id: \.element.key) { index, entry in
                        ESRComponentRow(name: ESRDisplay.label(for: entry.key), component: entry.value)
                        if index < esr.components.count - 1 {
                            Divider().overlay(Theme.Color.hairline)
                        }
                    }
                }
            }
        } else {
            Text("Representation and access evidence is unavailable for this trial.")
                .font(.subheadline)
                .foregroundStyle(Theme.Color.muted)
        }
    }
}

private struct ESRComponentRow: View {
    let name: String
    let component: ComponentEvidence
    @State private var showsRationale = false

    private var evidenceType: ESREvidenceType { ESREvidenceType(rawValue: component.evidence_type) }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name)
                        .font(.body)
                        .foregroundStyle(Theme.Color.ink)
                    Label(evidenceType.label, systemImage: evidenceType.symbolName)
                        .font(.caption)
                        .foregroundStyle(evidenceType.tintColor)
                }
                Spacer()
                Text(ScoreFormat.rounded(component.score))
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(component.score == nil ? Theme.Color.muted : Theme.Color.ink)
                    .accessibilityLabel(component.score == nil ? "Score unavailable" : "Score \(ScoreFormat.rounded(component.score))")
            }
            .padding(.vertical, Theme.Spacing.xs)

            Button {
                withAnimation(.easeInOut(duration: 0.2)) { showsRationale.toggle() }
            } label: {
                Text(showsRationale ? "Hide detail" : "Why this score?")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(Theme.Color.accent)
            }
            .buttonStyle(.plain)

            if showsRationale {
                VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                    Text(component.rationale)
                        .font(.caption)
                        .foregroundStyle(Theme.Color.ink)
                    if let source = component.source {
                        ProvenanceText(text: source)
                    }
                    ForEach(component.missing_evidence, id: \.self) { item in
                        Label(item, systemImage: "minus.circle")
                            .font(.caption)
                            .foregroundStyle(Theme.Color.muted)
                    }
                }
                .padding(.bottom, Theme.Spacing.s)
                .transition(.opacity)
            }
        }
        .padding(.vertical, Theme.Spacing.xs)
    }
}

#Preview {
    ESRScoreView(esr: ESRResult(
        score: 63,
        mode: "mixed",
        evidence_coverage: 0.6,
        components: [
            "socioeconomic": ComponentEvidence(score: 36, evidence_coverage: 1.0, evidence_type: "observed_geographic", rationale: "Site reach and geographic spread only.", source: "ClinicalTrials.gov site locations", missing_evidence: []),
            "sex": ComponentEvidence(score: 100, evidence_coverage: 1.0, evidence_type: "protocol_inclusivity", rationale: "Protocol eligibility is open to all sexes.", source: "eligibilityModule.sex", missing_evidence: []),
            "race": ComponentEvidence(score: nil, evidence_coverage: 0.0, evidence_type: "insufficient_data", rationale: "No reported participant race/ethnicity enrollment.", source: nil, missing_evidence: ["Participant race/ethnicity enrollment not reported."]),
        ],
        weights_used: ["socioeconomic": 0.35, "sex": 0.25, "race": 0.4]
    ))
    .padding()
}

#Preview("Unavailable") {
    ESRScoreView(esr: nil).padding()
}
