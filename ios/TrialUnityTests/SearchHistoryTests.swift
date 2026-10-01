import XCTest
@testable import TrialUnityIntegration

final class SearchHistoryTests: XCTestCase {
    private func response() throws -> TrialSearchResponse {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "recommendations", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(TrialSearchResponse.self, from: Data(contentsOf: url))
    }
    func testPersistenceDeduplicationAndProfileIsolation() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("history.json")
        let store = SearchHistoryStore(file: file)
        let profile = PatientProfile(draft: .sample)
        let data = try response()
        let id = UUID()
        store.record(id: id, profile: profile, response: data)
        store.record(id: id, profile: profile, response: data)
        XCTAssertEqual(store.records.count, 1)
        let restored = SearchHistoryStore(file: file)
        XCTAssertEqual(restored.records.first?.id, id)
        XCTAssertEqual(restored.records.first?.profile.condition, profile.condition)
        XCTAssertEqual(restored.records.first?.response.results.map(\.id), data.results.map(\.id))
        XCTAssertEqual(restored.records.first?.date, store.records.first?.date)
        XCTAssertTrue(SearchHistoryStore(file: directory.appendingPathComponent("other-user.json")).records.isEmpty)
        let next = UUID()
        restored.record(id: next, profile: profile, response: data)
        XCTAssertEqual(restored.records.map(\.id), [next, id])
    }
    @MainActor func testSnapshotOpensExistingResultsWithoutNewSearch() async throws {
        let data = try response()
        let record = SearchHistoryRecord(id: UUID(), date: Date(), profile: PatientProfile(draft: .sample), response: data)
        let model = MatchingModel(snapshot: record)
        await model.load()
        guard case .loaded(let loaded) = model.state else { return XCTFail("Expected saved results") }
        XCTAssertEqual(loaded.results.map(\.id), data.results.map(\.id))
        XCTAssertEqual(model.source, data.source)
        XCTAssertNil(model.historyRecord, "Opening a snapshot must not create a new history record")
    }
    func testWriteFailureDoesNotInventHistory() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = SearchHistoryStore(file: directory)
        store.record(id: UUID(), profile: PatientProfile(draft: .sample), response: try response())
        XCTAssertTrue(store.records.isEmpty)
        XCTAssertNotNil(store.message)
    }
}
