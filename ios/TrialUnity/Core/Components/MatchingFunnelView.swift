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

/// Shown while the request is in flight. Never fakes progress ticking or counts.
struct MatchingFunnelLoadingView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.l) {
            HStack(spacing: Theme.Spacing.s) {
                ProgressView()
                Text("Finding trials for you")
                    .font(.editorialTitle)
                    .foregroundStyle(Theme.Color.ink)
            }
            VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                ForEach(stageLabels, id: \.self) { stage in
                    HStack(spacing: Theme.Spacing.s) {
                        Image(systemName: "circle")
                            .foregroundStyle(Theme.Color.hairline)
                        Text(stage)
                            .font(.body)
                            .foregroundStyle(Theme.Color.ink)
                    }
                }
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Searching. Stages: \(stageLabels.joined(separator: ", "))")
        }
        .padding(Theme.Spacing.l)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Shown once results have loaded, with the real per-stage counts the backend
/// actually evaluated -- never invented.
struct MatchingFunnelSummaryView: View {
    let funnel: MatchingFunnel

    private var stages: [(label: String, count: Int)] {
        [
            ("Candidate studies", funnel.candidate_trials),
            ("Recruiting", funnel.recruiting_trials),
            ("Passed structured checks", funnel.structured_eligible_trials),
            ("Ranked matches", funnel.ranked_matches),
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.s) {
            ForEach(Array(stages.enumerated()), id: \.offset) { index, stage in
                HStack(spacing: Theme.Spacing.s) {
                    Text("\(stage.count)")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(Theme.Color.ink)
                        .frame(width: 40, alignment: .trailing)
                    Text(stage.label)
                        .font(.subheadline)
                        .foregroundStyle(Theme.Color.muted)
                    Spacer()
                }
                if index < stages.count - 1 {
                    Image(systemName: "arrow.down")
                        .font(.caption2)
                        .foregroundStyle(Theme.Color.hairline)
                        .padding(.leading, 12)
                }
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
