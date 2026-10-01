import SwiftUI

/// The results experience: a real retrieval-funnel summary followed by a
/// deliberate shortlist. Tapping a result pushes the full-screen Trial Passport.
struct MatchingView: View {
    @Environment(SearchHistoryStore.self) private var history
    @State private var model: MatchingModel
    @State private var attempt = 0
    @State private var openedResult: TrialRecommendation?
    @State private var openedSource = ""
    var onResultCount: (Int) -> Void = { _ in }

    init(model: MatchingModel, onResultCount: @escaping (Int) -> Void) {
        _model = State(initialValue: model)
        self.onResultCount = onResultCount
    }

    init(profile: PatientProfile) {
        _model = State(initialValue: MatchingModel(profile: profile))
    }

    private var isLoaded: Bool {
        if case .loaded = model.state { return true }
        return false
    }

    var body: some View {
        Group {
            switch model.state {
            case .idle, .loading:
                MatchingFunnelLoadingView(completedSteps: model.completedSteps)
            case .loaded(let response):
                resultsList(response)
            case .empty:
                emptyState
            case .failed(let error):
                failureState(error)
            }
        }
        .background(Theme.Color.paper)
        .toolbar {
            NavigationLink(destination: SavedTrialsView()) {
                Label("Saved", systemImage: "bookmark")
            }
        }
        .navigationDestination(isPresented: Binding(
            get: { openedResult != nil },
            set: { if !$0 { openedResult = nil } }
        )) {
            if let result = openedResult {
                TrialPassportView(profile: model.profile, result: result, responseSource: openedSource)
            }
        }
        .navigationTitle("Matching trials")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: attempt) {
            await model.load()
            if let record = model.historyRecord { history.record(id: record.id, profile: record.profile, response: record.response) }
            if case .loaded(let response) = model.state { onResultCount(response.results.count) }
            if case .empty = model.state { onResultCount(0) }
        }
        // A single, purposeful haptic when a search actually finishes -- not on
        // every re-render, and never implying more than "results are ready."
        .onChange(of: isLoaded) { _, loaded in
            if loaded { Haptics.matchingCompleted() }
        }
    }

    private func resultsList(_ response: TrialSearchResponse) -> some View {
        List {
            Section {
                ForEach(response.results) { result in
                    Button {
                        openedSource = response.source
                        openedResult = result
                    } label: {
                        TrialResultRow(result: result, preferredLocation: model.profile.location)
                    }
                    .buttonStyle(.pressable)
                    .accessibilityHint("Opens Trial Passport")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .listRowInsets(EdgeInsets(top: Theme.Spacing.xs, leading: Theme.Metrics.screenPadding, bottom: Theme.Spacing.xs, trailing: Theme.Metrics.screenPadding))
                    .swipeActions(edge: .leading) {
                        SaveTrialButton(result: result, profile: model.profile, source: response.source)
                    }
                }
            } header: {
                Text("Matches")
            }
            .listRowBackground(Color.clear)
            .listRowSeparator(.hidden)
        }
        .scrollContentBackground(.hidden)
        .listStyle(.plain)
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
            Text("No trials were found for this profile.")
                .font(.body)
                .foregroundStyle(Theme.Color.ink)
            PrimaryButton(title: "Retry", action: retry)
        }
        .padding(Theme.Metrics.screenPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func failureState(_ error: APIError) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
            Text(error.localizedDescription)
                .font(.body)
                .foregroundStyle(Theme.Color.ink)
            PrimaryButton(title: "Retry", action: retry)
        }
        .padding(Theme.Metrics.screenPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func retry() {
        model.prepareRetry()
        attempt += 1
    }
}

struct SearchHistoryView: View {
    @Environment(SearchHistoryStore.self) private var history
    var body: some View {
        List {
            if history.records.isEmpty {
                ContentUnavailableView("No searches yet", systemImage: "clock.arrow.circlepath",)
            }
            ForEach(history.records) { record in
                NavigationLink {
                    MatchingView(model: MatchingModel(snapshot: record), onResultCount: { _ in })
                } label: {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(record.profile.condition).font(.headline)
                        Text(record.date.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption).foregroundStyle(Theme.Color.muted)
                        Text("\(record.response.results.count) matches").font(.subheadline)
                    }.padding(.vertical, 6)
                }
            }
            if let message = history.message { Text(message).foregroundStyle(Theme.Color.attention) }
        }
        .navigationTitle("Search history")
    }
}
