import Foundation
import Observation

@MainActor @Observable
final class MatchingModel {
    enum State {
        case idle, loading, loaded(TrialSearchResponse), empty, failed(APIError)
    }
    private(set) var state: State = .idle
    private(set) var source: String?
    let profile: PatientProfile
    private let client: any RecommendationsProviding

    init(profile: PatientProfile, client: any RecommendationsProviding = APIClient()) {
        self.profile = profile
        self.client = client
    }

    func load() async {
        guard case .idle = state else { return }
        state = .loading
        do {
            let response = try await client.recommendations(for: profile)
            try Task.checkCancellation()
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
