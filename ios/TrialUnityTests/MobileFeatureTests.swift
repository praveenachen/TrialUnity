import XCTest
@testable import TrialUnityIntegration

final class MobileFeatureTests: XCTestCase {
    private func trial(_ id: String = "NCT1") throws -> SavedTrial {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "recommendations", withExtension: "json", subdirectory: "Fixtures"))
        let result = try JSONDecoder().decode(TrialSearchResponse.self, from: Data(contentsOf: url)).results[0]
        return SavedTrial(id: id, title: "Study \(id)", savedAt: nil, source: "sample-data", profile: PatientProfile(draft: .sample), recommendation: result)
    }
    private func file() -> URL { FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("state.json") }

    func testSelectionLimitAndDuplicates() throws {
        let one = try trial()
        XCTAssertNil(AppointmentSession(trials: [], context: ""))
        XCTAssertEqual(AppointmentSession(trials: [one, one], context: "")?.selectedIDs, [one.id])
        XCTAssertEqual(AppointmentSession(trials: try (1...3).map { try trial("NCT\($0)") }, context: "")?.trials.count, 3)
        XCTAssertNil(AppointmentSession(trials: try (1...4).map { try trial("NCT\($0)") }, context: ""))
    }
    func testPersistenceRestorationAndCompletion() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let records = try [trial(), trial("NCT2")]
        let store = AppointmentStore(file: path)
        XCTAssertTrue(store.start(records, context: "My appointment"))
        let question = try XCTUnwrap(AppointmentBrief.discussionItems(for: records[0]).first(where: \.isQuestion))
        store.toggle(question, trialID: records[0].id)
        store.setNote("Ask about travel\nBring list", trialID: records[0].id)
        store.reviewCurrent(); store.move(to: 1)
        let restored = AppointmentStore(file: path)
        XCTAssertEqual(restored.session?.currentIndex, 1)
        XCTAssertEqual(restored.session?.selectedIDs, records.map(\.id))
        XCTAssertEqual(restored.session?.notes[records[0].id], "Ask about travel\nBring list")
        XCTAssertEqual(restored.session?.discussedQuestionCount, 1)
        XCTAssertEqual(restored.session?.noteCount, 1)
        restored.complete()
        let completed = try XCTUnwrap(AppointmentStore(file: path).session)
        XCTAssertTrue(completed.completed)
        XCTAssertEqual(completed.reviewedIDs.count, 2)
        XCTAssertEqual(AppointmentBrief.summary(completed), AppointmentBrief.summary(completed))
        XCTAssertTrue(AppointmentBrief.summary(completed).contains("Open items:"))
        XCTAssertTrue(AppointmentBrief.summary(completed).contains("Discussed items:"))
        XCTAssertTrue(AppointmentBrief.summary(completed).contains(records[0].sourceLink))
        XCTAssertTrue(AppointmentBrief.summary(completed).contains(AppointmentBrief.disclaimer))
        XCTAssertEqual(AppointmentBrief.summary(completed), AppointmentBrief.summary(try JSONDecoder().decode(AppointmentSession.self, from: JSONEncoder().encode(completed))))
    }
    func testDiscussedNeverChangesEligibilityOrEvidence() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let record = try trial()
        let store = AppointmentStore(file: path)
        store.start([record], context: "")
        let before = AppointmentBrief.generate([record], context: "")
        for item in AppointmentBrief.discussionItems(for: record) { store.toggle(item, trialID: record.id) }
        let after = try XCTUnwrap(store.session)
        XCTAssertEqual(AppointmentBrief.generate(after.trials, context: ""), before)
        XCTAssertEqual(after.trials[0].recommendation?.structured_eligibility?.status, record.recommendation?.structured_eligibility?.status)
        XCTAssertTrue(AppointmentBrief.summary(after).contains("does not mean eligible, confirmed, or resolved"))
        XCTAssertFalse(after.completed)
        store.move(to: 20)
        XCTAssertEqual(store.session?.currentIndex, 0)
    }
    func testCannotCompleteUnreviewedTrialsAndIgnoresUnknownItems() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let records = try [trial(), trial("NCT2")]
        let store = AppointmentStore(file: path)
        store.start(records, context: "")
        store.move(to: 1); store.complete()
        XCTAssertFalse(try XCTUnwrap(store.session).completed)
        store.toggle(AppointmentItem(id: "invalid", text: "Invented", isQuestion: true), trialID: "NCT2")
        XCTAssertEqual(store.session?.discussedQuestionCount, 0)
        store.setNote("   \n", trialID: "NCT2")
        XCTAssertEqual(store.session?.noteCount, 0)
    }
    func testCorruptSessionAndWriteFailureAreReported() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("broken".utf8).write(to: path)
        XCTAssertNil(AppointmentStore(file: path).session)
        XCTAssertNotNil(AppointmentStore(file: path).message)
        let store = AppointmentStore(file: path.appendingPathComponent("cannot-write.json"))
        XCTAssertFalse(store.start([try trial()], context: ""))
        XCTAssertNil(store.session)
        XCTAssertNotNil(store.message)
    }
    @MainActor func testGeocodeCacheSuccessMissAndInvalidCoordinates() async throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let cache = GeocodeCache(file: path)
        let point = SiteCoordinate(latitude: 43.65, longitude: -79.38, precise: false)
        var requests = 0
        let first = await cache.resolve(" Toronto, Ontario ") { _ in requests += 1; return point }
        XCTAssertEqual(first, point)
        let second = await GeocodeCache(file: path).resolve("toronto, ontario") { _ in requests += 1; return nil }
        XCTAssertEqual(second, point)
        XCTAssertEqual(requests, 1)
        let missing = await cache.resolve("Unresolved") { _ in nil }
        XCTAssertNil(missing)
        let retry = await cache.resolve("Unresolved") { _ in point }
        XCTAssertEqual(retry, point)
        let invalid = await cache.resolve("Invalid") { _ in SiteCoordinate(latitude: 100, longitude: 0, precise: true) }
        XCTAssertNil(invalid)
        let blank = await cache.resolve("  ") { _ in XCTFail("Must not geocode empty input"); return point }
        XCTAssertNil(blank)
    }
    func testLegacyLocationFallbackAndNearestReliability() throws {
        let record = try trial()
        let trial = try XCTUnwrap(record.recommendation?.trial)
        XCTAssertEqual(trial.mapSites.map(\.location), trial.locations)
        let origin = SiteCoordinate(latitude: 43.65, longitude: -79.38, precise: true)
        let site = TrialSite(facility: "Test facility", location: "Toronto", status: "RECRUITING", latitude: 43.66, longitude: -79.38)
        let resolved = ResolvedTrialSite(id: 0, site: site, coordinate: site.coordinate)
        XCTAssertEqual(SiteDistance.nearest([resolved], origin: origin)?.id, 0)
        XCTAssertGreaterThan(origin.meters(to: try XCTUnwrap(site.coordinate)), 1000)
        XCTAssertNil(SiteDistance.nearest([resolved], origin: .init(latitude: 43.65, longitude: -79.38, precise: false)))
        XCTAssertNil(SiteDistance.nearest([resolved, .init(id: 1, site: site, coordinate: nil)], origin: origin))
        XCTAssertNil(SiteDistance.nearest([.init(id: 0, site: site, coordinate: .init(latitude: 43.66, longitude: -79.38, precise: false))], origin: origin))
        XCTAssertNil(TrialSite(facility: nil, location: "Invalid", status: nil, latitude: 91, longitude: 0).coordinate)
    }
}
