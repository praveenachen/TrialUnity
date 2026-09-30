import SwiftUI

struct ESRScoreView: View {
    let esr: ESRResult?
    var body: some View {
        BrandedSurface {
            VStack(alignment: .leading, spacing: 16) {
                Text("Representation & access").font(.headline)
                if let esr {
                    HStack(alignment: .firstTextBaseline) {
                        Text(ScoreFormat.rounded(esr.score)).font(.heroNumber)
                        Text("/ 100").foregroundStyle(Theme.Color.muted)
                    }.accessibilityElement(children: .combine)
                    Text(PatientPresentation.evidence(esr.evidence_coverage)).font(.subheadline.weight(.semibold))
                    CoverageBar(coverage: esr.evidence_coverage, label: "Evidence available")
                    GlanceGrid(items: [
                        ("Access", ScoreFormat.rounded(esr.components["socioeconomic"]?.score)),
                        ("Inclusivity", ScoreFormat.rounded(esr.components["sex"]?.score)),
                        ("Race data", ScoreFormat.rounded(esr.components["race"]?.score))
                    ])
                    DisclosureGroup("Understand this score") {
                        VStack(alignment: .leading, spacing: 16) {
                            Text("This describes available representation and access evidence, not your medical eligibility. Missing evidence is not a zero score.")
                            Text("Evidence mode: \(esr.mode.replacingOccurrences(of: "_", with: " "))")
                            ForEach(ESRDisplay.orderedComponents(esr.components), id: \.key) { entry in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(ESRDisplay.label(for: entry.key)).font(.headline)
                                    Text(ScoreFormat.rounded(entry.value.score))
                                    Text(ESREvidenceType(rawValue: entry.value.evidence_type).label)
                                    Text(entry.value.rationale)
                                    CoverageBar(coverage: entry.value.evidence_coverage)
                                    if let source = entry.value.source { Text("Source: \(source)") }
                                    ForEach(entry.value.missing_evidence, id: \.self) { Text($0) }
                                }
                            }
                            DisclosureGroup("Calculation & provenance") {
                                Text("The backend combines available components using renormalized weights. Evidence coverage reports completeness separately; it does not increase the score.")
                                ForEach(esr.weights_used.keys.sorted(), id: \.self) { key in
                                    Text("\(ESRDisplay.label(for: key)): \(ScoreFormat.clamped(esr.weights_used[key] ?? 0).formatted(.percent))")
                                }
                            }
                        }.font(.footnote).padding(.top, 12)
                    }
                } else {
                    Text("—").font(.heroNumber)
                    Text("Evidence not reported").foregroundStyle(Theme.Color.muted)
                }
            }
        }
    }
}
