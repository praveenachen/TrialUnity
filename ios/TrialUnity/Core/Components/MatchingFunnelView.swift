import SwiftUI

/// The four conceptual retrieval stages. Shared between the loading state (no
/// counts yet -- the backend does this in one request, so there is no honest
/// per-stage progress to animate) and the loaded state (real counts from
/// `MatchingFunnel`).
private let stageLabels = [
    "Searching current studies",
    "Checking structured eligibility",
    "Ranking relevant trials",
    "Reviewing representation evidence",
]

/// Shown while the request is in flight. Never fakes progress ticking or counts:
/// the backend answers in one request, so there's no real per-stage completion
/// signal to report. Instead, a single highlight drifts down the stage list on a
/// fixed cadence -- motion that says "still working," not "stage N is done."
/// Skips the drift (shows a plain static list) when Reduce Motion is on.
struct MatchingFunnelLoadingView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 28) {
            DecisionAnchor(symbol: "point.3.connected.trianglepath.dotted", caption: "From study records to a shortlist you can explore.")
            Text("Finding trials for you").font(.title.bold())
            ProgressView("Searching and reviewing evidence…")
            Text("Counts appear when your search finishes.").font(.subheadline).foregroundStyle(Theme.Color.muted)
            VStack(alignment: .leading, spacing: 12) {
                ForEach(stageLabels, id: \.self) { stage in
                    Label(stage, systemImage: "circle").font(.subheadline).foregroundStyle(Theme.Color.muted)
                }
            }
        }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Shown once results have loaded, with the real per-stage counts the backend
/// actually evaluated -- never invented. One compact, branded module, not four
/// plain floating rows: quick comprehension, not analytics.
struct MatchingFunnelSummaryView: View {
    let funnel: MatchingFunnel

    private var stages: [(count: Int, label: String)] {
        [
            (funnel.candidate_trials, "Studies found"),
            (funnel.recruiting_trials, "Recruiting"),
            (funnel.structured_eligible_trials, "Structured checks"),
            (funnel.ranked_matches, "Matches"),
        ]
    }

    var body: some View {
        BrandedSurface {
            VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                Text("YOUR SEARCH")
                    .font(.sectionLabel)
                    .foregroundStyle(Theme.Color.muted)
                    .tracking(0.5)

                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(stages.enumerated()), id: \.offset) { index, stage in
                        HStack {
                            Text("\(stage.count)").font(.title3.bold()).monospacedDigit().frame(minWidth: 44, alignment: .leading)
                            Text(stage.label).font(.subheadline)
                        }
                        if index < stages.count - 1 { Image(systemName: "arrow.down").font(.caption).foregroundStyle(Theme.Color.accent).padding(.leading, 14) }
                    }
                }

                Capsule()
                    .fill(Theme.Color.hairline)
                    .frame(height: 3)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(stages.map { "\($0.label): \($0.count)" }.joined(separator: ". "))
    }
}

#Preview("Loading") {
    MatchingFunnelLoadingView()
}

#Preview("Summary") {
    MatchingFunnelSummaryView(funnel: MatchingFunnel(
        candidate_trials: 25, recruiting_trials: 25, structured_eligible_trials: 18, ranked_matches: 10
    ))
    .padding()
}
