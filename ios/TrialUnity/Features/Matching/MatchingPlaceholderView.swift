import SwiftUI

/// A native placeholder for the future retrieval flow. This does not call the
/// backend, does not simulate progress ticking off, and does not fabricate any
/// counts -- it is honestly a preview of what will happen once matching is wired
/// up to the API in a later phase.
struct MatchingPlaceholderView: View {
    private let plannedSteps = [
        "Searching current studies",
        "Checking structured eligibility",
        "Ranking relevant trials",
        "Reviewing representation evidence",
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.l) {
                VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                    Text("Finding trials for you")
                        .font(.editorialTitle)
                        .foregroundStyle(Theme.Color.ink)
                    Text("This is a preview of what happens next. TrialUnity isn't connected to live trial data yet.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.Color.muted)
                }

                VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                    ForEach(plannedSteps, id: \.self) { step in
                        HStack(spacing: Theme.Spacing.s) {
                            Image(systemName: "circle")
                                .foregroundStyle(Theme.Color.hairline)
                            Text(step)
                                .font(.body)
                                .foregroundStyle(Theme.Color.ink)
                        }
                    }
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Planned matching steps: \(plannedSteps.joined(separator: ", "))")

                Text("SOURCE: CLINICALTRIALS.GOV")
                    .font(.provenance)
                    .foregroundStyle(Theme.Color.muted)
            }
            .padding(Theme.Spacing.l)
        }
        .background(Theme.Color.paper)
        .navigationTitle("Matching")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack {
        MatchingPlaceholderView()
    }
}
