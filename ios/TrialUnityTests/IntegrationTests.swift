import XCTest
@testable import TrialUnityIntegration

final class IntegrationTests: XCTestCase {
    func fixture() throws -> Data {
        try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: "recommendations", withExtension: "json", subdirectory: "Fixtures")))
    }

    func testProfileEncoding() throws {
        let draft = PatientProfileDraft.sample
        draft.sex = .preferNotToSay
        draft.location = "  "
        draft.ageText = ""
        let data = try JSONEncoder().encode(PatientProfile(draft: draft))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(json["sex"])
        XCTAssertNil(json["age"])
        XCTAssertNil(json["location"])
        XCTAssertNil(json["travelPreference"])
        XCTAssertEqual(Set(json.keys), ["condition", "intervention_preferences", "phase_preferences", "notes"])
        XCTAssertEqual(json["notes"] as? String, draft.notes)
        XCTAssertEqual(json["intervention_preferences"] as? [String], draft.interventionPreferences)
        draft.sex = .female
        XCTAssertEqual(PatientProfile(draft: draft).sex, "Female")
        draft.condition = " a "
        XCTAssertFalse(draft.isConditionValid)
    }

    func testContractDecodingAndNullableEvidence() throws {
        let response = try JSONDecoder().decode(TrialSearchResponse.self, from: fixture())
        let result = try XCTUnwrap(response.results.first)
        XCTAssertEqual(response.source, "sample-data")
        XCTAssertEqual(result.relevance.overall, 0.6)
        XCTAssertEqual(result.structured_eligibility?.status, "unknown")
        XCTAssertNil(result.trial.enrollment_race_distribution)
        XCTAssertNil(result.esr?.score)
        XCTAssertNil(result.esr?.components["race"]?.score)
        XCTAssertEqual(result.representation_risk?.evidence_type, "predicted")
        XCTAssertEqual(result.representation_risk?.probabilities?["moderate"], 0.6)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture()) as? [String: Any])
        var results = try XCTUnwrap(json["results"] as? [[String: Any]])
        var risk = try XCTUnwrap(results[0]["representation_risk"] as? [String: Any])
        risk["risk_level"] = NSNull()
        risk["confidence"] = NSNull()
        risk["probabilities"] = NSNull()
        results[0]["representation_risk"] = risk
        json["results"] = results
        let unknownRisk = try JSONDecoder().decode(TrialSearchResponse.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(unknownRisk.results[0].representation_risk?.risk_level)
        XCTAssertNil(unknownRisk.results[0].representation_risk?.confidence)
        XCTAssertNil(unknownRisk.results[0].representation_risk?.probabilities)
        results[0]["esr"] = NSNull()
        results[0]["representation_risk"] = NSNull()
        json["results"] = results
        let nullable = try JSONDecoder().decode(TrialSearchResponse.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(nullable.results[0].esr)
        XCTAssertNil(nullable.results[0].representation_risk)
    }

    func testHTTPAndTransportErrors() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let client = APIClient(configuration: .init(baseURL: URL(string: "http://127.0.0.1:8001")), session: URLSession(configuration: config))
        let profile = PatientProfile(draft: .sample)
        for (status, body, expected) in [
            (422, "{\"detail\":[{\"msg\":\"Too short\"}]}", APIError.http(status: 422, detail: "Too short")),
            (503, "{\"detail\":\"Unavailable\"}", APIError.http(status: 503, detail: "Unavailable")),
            (200, "{}", APIError.malformedResponse)
        ] {
            StubURLProtocol.handler = { request in
                XCTAssertEqual(request.httpMethod, "POST")
                XCTAssertEqual(request.url?.path, "/api/recommendations")
                return (status, Data(body.utf8))
            }
            do { _ = try await client.recommendations(for: profile); XCTFail("Expected failure") }
            catch { XCTAssertEqual(error as? APIError, expected) }
        }
        for (code, expected) in [(URLError.timedOut, APIError.timeout), (.cannotConnectToHost, .unavailable)] {
            StubURLProtocol.handler = { _ in throw URLError(code) }
            do { _ = try await client.recommendations(for: profile); XCTFail("Expected failure") }
            catch { XCTAssertEqual(error as? APIError, expected) }
        }
        let data = try fixture()
        StubURLProtocol.handler = { _ in (200, data) }
        let response = try await client.recommendations(for: profile)
        XCTAssertEqual(response.results.count, 1)
    }

    @MainActor func testStreamProgressAndResult() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        let client = APIClient(configuration: .init(baseURL: URL(string: "http://127.0.0.1:8001")), session: URLSession(configuration: configuration))
        let responseJSON = String(decoding: try fixture(), as: UTF8.self).replacingOccurrences(of: "\n", with: "")
        let events = "{\"completed\":1}\n{\"completed\":2}\n{\"completed\":3,\"response\":\(responseJSON)}\n"
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/recommendations/stream")
            return (200, Data(events.utf8))
        }
        var progress: [Int] = []
        let response = try await client.recommendations(for: PatientProfile(draft: .sample)) { progress.append($0) }
        XCTAssertEqual(progress, [1, 2, 3])
        XCTAssertEqual(response.results.count, 1)
    }

    @MainActor func testStreamFailureDoesNotCompleteRemainingChecks() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        let client = APIClient(configuration: .init(baseURL: URL(string: "http://127.0.0.1:8001")), session: URLSession(configuration: configuration))
        StubURLProtocol.handler = { _ in (200, Data("{\"completed\":1}\n{\"error\":\"Search failed\"}\n".utf8)) }
        var progress: [Int] = []
        do {
            _ = try await client.recommendations(for: PatientProfile(draft: .sample)) { progress.append($0) }
            XCTFail("Expected failure")
        } catch { XCTAssertEqual(error as? APIError, .unavailable) }
        XCTAssertEqual(progress, [1])
    }

    @MainActor func testStateTransitionsAndDuplicateGuard() async throws {
        let client = ControlledClient()
        let model = MatchingModel(profile: PatientProfile(draft: .sample), client: client)
        guard case .idle = model.state else { return XCTFail() }
        let task = Task { await model.load() }
        while client.continuation == nil { await Task.yield() }
        guard case .loading = model.state else { return XCTFail() }
        await model.load()
        XCTAssertEqual(client.calls, 1)
        let response = try JSONDecoder().decode(TrialSearchResponse.self, from: fixture())
        client.continuation?.resume(returning: response)
        await task.value
        guard case .loaded = model.state else { return XCTFail() }
        await model.load()
        XCTAssertEqual(client.calls, 1)

        let failing = MatchingModel(profile: PatientProfile(draft: .sample), client: ImmediateClient(result: .failure(APIError.timeout)))
        await failing.load()
        guard case .failed(.timeout) = failing.state else { return XCTFail() }
        failing.prepareRetry()
        guard case .idle = failing.state else { return XCTFail() }
        await failing.load()
        guard case .failed = failing.state else { return XCTFail() }
        let emptyFunnel = MatchingFunnel(candidate_trials: 0, recruiting_trials: 0, structured_eligible_trials: 0, ranked_matches: 0)
        let empty = MatchingModel(profile: PatientProfile(draft: .sample), client: ImmediateClient(result: .success(TrialSearchResponse(query: "cancer", total: 0, results: [], source: "sample-data", funnel: emptyFunnel))))
        await empty.load()
        guard case .empty = empty.state else { return XCTFail() }
        let cancelled = MatchingModel(profile: PatientProfile(draft: .sample), client: ImmediateClient(result: .failure(CancellationError())))
        await cancelled.load()
        guard case .idle = cancelled.state else { return XCTFail() }
    }

    @MainActor func testChecklistAdvancesAndCompletesOnlyOnSuccess() async throws {
        let response = try JSONDecoder().decode(TrialSearchResponse.self, from: fixture())
        let ok = MatchingModel(profile: PatientProfile(draft: .sample), client: SteppedClient(result: .success(response)),
                               paceInterval: .milliseconds(10), completionPause: .milliseconds(1))
        let task = Task { await ok.load() }
        while ok.completedSteps < 3 { await Task.yield() }   // paced past stage 1, still in flight
        XCTAssertLessThan(ok.completedSteps, MatchingModel.stepCount)
        await task.value
        XCTAssertEqual(ok.completedSteps, MatchingModel.stepCount)

        let bad = MatchingModel(profile: PatientProfile(draft: .sample), client: SteppedClient(result: .failure(APIError.unavailable)),
                                paceInterval: .milliseconds(10), completionPause: .milliseconds(1))
        await bad.load()
        guard case .failed = bad.state else { return XCTFail() }
        XCTAssertLessThan(bad.completedSteps, MatchingModel.stepCount)
    }

    func testLocalEndToEnd() async throws {
        guard ProcessInfo.processInfo.environment["TRIALUNITY_RUN_E2E"] == "1" else {
            throw XCTSkip("Set TRIALUNITY_RUN_E2E=1 with FastAPI running on port 8001")
        }
        let response = try await APIClient().recommendations(for: PatientProfile(draft: .sample))
        XCTAssertEqual(response.query, PatientProfileDraft.sample.condition)
        XCTAssertEqual(response.total, response.results.count)
        XCTAssertFalse(response.source.isEmpty)
        print("FastAPI end-to-end: source=\(response.source), recommendations=\(response.total)")
    }
}

private final class StubURLProtocol: URLProtocol {
    static var handler: ((URLRequest) throws -> (Int, Data))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status, data) = try Self.handler!(request)
            client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

private final class ControlledClient: RecommendationsProviding {
    var calls = 0
    var continuation: CheckedContinuation<TrialSearchResponse, Error>?
    func recommendations(for profile: PatientProfile) async throws -> TrialSearchResponse {
        calls += 1
        return try await withCheckedThrowingContinuation { continuation = $0 }
    }
}
private struct ImmediateClient: RecommendationsProviding {
    let result: Result<TrialSearchResponse, Error>
    func recommendations(for profile: PatientProfile) async throws -> TrialSearchResponse { try result.get() }
}

/// Reports the first backend stage, then stays in flight briefly before resolving.
private struct SteppedClient: RecommendationsProviding {
    let result: Result<TrialSearchResponse, Error>
    func recommendations(for profile: PatientProfile) async throws -> TrialSearchResponse { try result.get() }
    func recommendations(for profile: PatientProfile, progress: @escaping @MainActor (Int) -> Void) async throws -> TrialSearchResponse {
        await progress(1)
        try await Task.sleep(for: .milliseconds(300))
        return try result.get()
    }
}
