import Foundation
import Observation

@MainActor @Observable
final class MatchingModel {
    enum State {
        case idle, loading, loaded(TrialSearchResponse), empty, failed(APIError)
    }
    private(set) var state: State = .idle
    private(set) var source: String?
    private(set) var historyRecord: SearchHistoryRecord?
    /// Checklist steps finished so far, 0...stepCount. Reaches stepCount only once the request succeeds.
    private(set) var completedSteps = 0
    let profile: PatientProfile
    private let client: any RecommendationsProviding
    private let paceInterval: Duration
    private let completionPause: Duration

    static let stepCount = 4
    /// While the request is in flight the last step is at most "current", never done.
    private static let lastActiveStep = stepCount - 1

    init(profile: PatientProfile, client: any RecommendationsProviding = APIClient(),
         paceInterval: Duration = .milliseconds(1600), completionPause: Duration = .milliseconds(450)) {
        self.profile = profile
        self.client = client
        self.paceInterval = paceInterval
        self.completionPause = completionPause
    }

    convenience init(snapshot: SearchHistoryRecord) {
        self.init(profile: snapshot.profile)
        source = snapshot.response.source
        completedSteps = Self.stepCount
        state = snapshot.response.results.isEmpty ? .empty : .loaded(snapshot.response)
    }

    /// Maps the backend's real stages to checklist steps. Stage 1 = studies found;
    /// stage 2 = eligibility, ranking and evidence scoring finished server-side.
    private static func steps(forBackendStage stage: Int) -> Int {
        switch stage {
        case ..<1: return 0
        case 1: return 1
        default: return lastActiveStep
        }
    }

    func load() async {
        guard case .idle = state else { return }
        state = .loading
        completedSteps = 0
        // The backend reports one stage boundary covering eligibility, ranking and
        // evidence review, so once the study search is really done the checklist
        // paces through those steps client-side. It never passes the last step
        // and never marks anything done on its own before the request succeeds.
        let pacer = Task { [weak self, paceInterval] in
            while !Task.isCancelled {
                try? await Task.sleep(for: paceInterval)
                guard !Task.isCancelled, let self else { return }
                if self.completedSteps >= 1, self.completedSteps < Self.lastActiveStep { self.completedSteps += 1 }
            }
        }
        defer { pacer.cancel() }
        do {
            let response = try await client.recommendations(for: profile) { [weak self] count in
                guard let self else { return }
                self.completedSteps = max(self.completedSteps, Self.steps(forBackendStage: count))
            }
            try Task.checkCancellation()
            pacer.cancel()
            completedSteps = Self.stepCount
            try? await Task.sleep(for: completionPause)
            try Task.checkCancellation()
            historyRecord = SearchHistoryRecord(id: UUID(), date: Date(), profile: profile, response: response)
            source = response.source
            state = response.results.isEmpty ? .empty : .loaded(response)
        } catch is CancellationError {
            state = .idle
        } catch {
            state = .failed(APIError.map(error))
        }
    }

    func prepareRetry() {
        switch state {
        case .failed, .empty: state = .idle
        default: break
        }
    }
}
