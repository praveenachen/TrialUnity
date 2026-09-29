import SwiftUI

/// The results experience: a real retrieval-funnel summary followed by a
/// deliberate shortlist. Tapping a result pushes the full-screen Trial Passport.
struct MatchingView: View {
    @State private var model: MatchingModel
    @State private var attempt = 0

    init(profile: PatientProfile) {
        _model = State(initialValue: MatchingModel(profile: profile))
    }

    var body: some View {
        Group {
            switch model.state {
            case .idle, .loading:
                ScrollView { MatchingFunnelLoadingView() }
            case .loaded(let response):
                resultsList(response)
            case .empty:
                emptyState
            case .failed(let error):
                failureState(error)
            }
        }
        .background(Theme.Color.paper)
        .navigationTitle("Matching trials")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: attempt) { await model.load() }
    }

    private func resultsList(_ response: TrialSearchResponse) -> some View {
        List {
            Section {
                MatchingFunnelSummaryView(funnel: response.funnel)
            }
            .listRowBackground(Theme.Color.paper)
            .listRowSeparator(.hidden)

            Section("Matches") {
                ForEach(response.results) { result in
                    NavigationLink {
                        TrialPassportView(profile: model.profile, result: result, responseSource: response.source)
                    } label: {
                        TrialResultRow(result: result)
                    }
                }
            }
            #if DEBUG
            Section("Development provenance") {
                Text("API: \(APIConfiguration.current.baseURL?.absoluteString ?? "Not configured")")
                Text("Source: \(response.source)")
            }
            .font(.caption)
            .foregroundStyle(Theme.Color.muted)
            #endif
        }
        .scrollContentBackground(.hidden)
        .listStyle(.plain)
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
            Text("No trials were found for this profile.")
                .font(.body)
                .foregroundStyle(Theme.Color.ink)
            Text("You can edit your profile or retry.")
                .font(.subheadline)
                .foregroundStyle(Theme.Color.muted)
            PrimaryButton(title: "Retry", action: retry)
        }
        .padding(Theme.Spacing.l)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func failureState(_ error: APIError) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
            Text(error.localizedDescription)
                .font(.body)
                .foregroundStyle(Theme.Color.ink)
            PrimaryButton(title: "Retry", action: retry)
        }
        .padding(Theme.Spacing.l)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func retry() {
        model.prepareRetry()
        attempt += 1
    }
}
