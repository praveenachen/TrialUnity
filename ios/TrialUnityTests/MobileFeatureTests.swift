import XCTest
import CoreLocation
@testable import TrialUnityIntegration

final class MobileFeatureTests: XCTestCase {
    // CoreLocation preserves the iOS status raw value on the macOS test host.
    private var whenInUse: CLAuthorizationStatus { CLAuthorizationStatus(rawValue: 4)! }
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
        let summary = AppointmentBrief.summary(completed)
        XCTAssertTrue(summary.contains("Discussed items:\n- \(question.text)\nOpen items:"))
        XCTAssertTrue(summary.contains("User notes:\nAsk about travel\nBring list"))
        XCTAssertTrue(summary.hasSuffix(AppointmentBrief.generate(records, context: "My appointment")))
        XCTAssertEqual(completed.context, "My appointment")
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
    func testInvalidRestoredIndexPreservesFileAndRejectedStartPreservesSession() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let store = AppointmentStore(file: path)
        XCTAssertTrue(store.start([try trial()], context: "Keep this"))
        XCTAssertFalse(store.start(try (1...4).map { try trial("NCT\($0)") }, context: "Replace"))
        XCTAssertEqual(store.session?.context, "Keep this")
        XCTAssertEqual(AppointmentStore(file: path).session?.context, "Keep this")
        var invalid = try XCTUnwrap(store.session)
        invalid.currentIndex = 99
        let data = try JSONEncoder().encode(invalid)
        try data.write(to: path)
        let restored = AppointmentStore(file: path)
        XCTAssertNil(restored.session)
        XCTAssertNotNil(restored.message)
        XCTAssertEqual(try Data(contentsOf: path), data)
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
    @MainActor func testLocationPermissionIsExplicitAndDenialKeepsSites() async throws {
        var permissions = 0
        var requests = 0
        let profile = SiteCoordinate(latitude: 43.65, longitude: -79.38, precise: false)
        let model = TrialLocationModel(lookup: { _ in profile },
            requestAuthorization: { _ in permissions += 1 }, requestLocation: { _ in requests += 1 })
        await model.load(trial: try XCTUnwrap(trial().recommendation?.trial), profileLocation: "Toronto")
        model.authorizationChanged(whenInUse)
        XCTAssertEqual(permissions, 0)
        XCTAssertEqual(requests, 0)
        model.useCurrentLocation(status: .notDetermined)
        XCTAssertEqual(permissions, 1)
        XCTAssertTrue(model.locating)
        model.authorizationChanged(.denied)
        XCTAssertFalse(model.locating)
        XCTAssertEqual(model.origin, profile)
        XCTAssertNotNil(model.message)
        XCTAssertFalse(model.sites.isEmpty)
        XCTAssertNil(model.nearest)
        model.useCurrentLocation(status: .restricted)
        XCTAssertEqual(requests, 0)
        XCTAssertEqual(model.origin, profile)
        model.useCurrentLocation(status: whenInUse)
        XCTAssertEqual(requests, 1)
    }

    @MainActor func testLocationFailureAccuracyAndLateCallbacks() {
        let model = TrialLocationModel(requestAuthorization: { _ in }, requestLocation: { _ in })
        let manager = CLLocationManager()
        model.useCurrentLocation(status: whenInUse)
        model.locationManager(manager, didFailWithError: CLError(.locationUnknown))
        XCTAssertFalse(model.locating)
        XCTAssertNotNil(model.message)
        for (accuracy, age) in [(-1.0, 0.0), (2000.0, 0.0), (20.0, -180.0)] {
            model.useCurrentLocation(status: whenInUse)
            model.locationManager(manager, didUpdateLocations: [CLLocation(
                coordinate: .init(latitude: 43.65, longitude: -79.38), altitude: 0,
                horizontalAccuracy: accuracy, verticalAccuracy: 0, timestamp: Date().addingTimeInterval(age))])
            XCTAssertNil(model.origin)
            XCTAssertFalse(model.locating)
            XCTAssertNotNil(model.message)
        }
        let valid = CLLocation(latitude: 43.65, longitude: -79.38)
        model.locationManager(manager, didUpdateLocations: [valid])
        XCTAssertEqual(model.origin?.precise, true)
        XCTAssertNil(model.message)
        model.useProfileLocation("Toronto")
        model.locationManager(manager, didFailWithError: CLError(.locationUnknown))
        model.locationManager(manager, didUpdateLocations: [valid])
        model.authorizationChanged(.denied)
        XCTAssertNil(model.origin)
        XCTAssertNil(model.message)
        XCTAssertEqual(model.originLabel, "Profile: Toronto")
    }

    @MainActor func testFacilityGeocodeFallsBackToCityAndPreservesUnresolvedSites() async throws {
        var value = try XCTUnwrap(trial().recommendation?.trial)
        value.trial_sites = [
            TrialSite(facility: "Unresolved facility", location: "Toronto", status: nil, latitude: nil, longitude: nil),
            TrialSite(facility: nil, location: "Unknown", status: nil, latitude: nil, longitude: nil)
        ]
        var addresses: [String] = []
        let model = TrialLocationModel(lookup: { address in
            addresses.append(address)
            return address == "Toronto" ? SiteCoordinate(latitude: 43.65, longitude: -79.38, precise: false) : nil
        })
        await model.load(trial: value, profileLocation: nil)
        XCTAssertEqual(addresses, ["Unresolved facility, Toronto", "Toronto", "Unknown"])
        XCTAssertEqual(model.sites.count, 2)
        XCTAssertEqual(model.sites[0].coordinate?.precise, false)
        XCTAssertNil(model.sites[1].coordinate)
        XCTAssertFalse(model.loading)
        await model.load(trial: value, profileLocation: nil)
        XCTAssertEqual(addresses.count, 3)
    }

    func testAppointmentHistoryPreservesDatesAndIndependentRecords() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let store = AppointmentStore(file: path)
        XCTAssertTrue(store.start([try trial()], context: "First visit"))
        let first = try XCTUnwrap(store.records.first)
        XCTAssertNotNil(first.createdAt)
        store.setNote("First notes", trialID: "NCT1")
        store.complete()
        let completedAt = try XCTUnwrap(store.records.first?.completedAt)
        XCTAssertTrue(store.start([try trial("NCT2")], context: "Second visit"))
        XCTAssertEqual(store.records.count, 2)
        XCTAssertEqual(store.records.first?.completedAt, completedAt)
        XCTAssertEqual(store.records.first?.session.notes["NCT1"], "First notes")
        store.reopen(recordID: first.id)
        store.setNote("Updated first notes", trialID: "NCT1", recordID: first.id)
        store.setContext("Updated visit context", recordID: first.id)
        XCTAssertEqual(store.session?.context, "Second visit")
        XCTAssertTrue(try XCTUnwrap(store.session).notes.isEmpty)
        XCTAssertNil(store.record(first.id)?.completedAt)
        store.complete(recordID: first.id)
        let restored = AppointmentStore(file: path)
        XCTAssertEqual(restored.records.map(\.id), store.records.map(\.id))
        XCTAssertEqual(restored.record(first.id)?.createdAt, first.createdAt)
        XCTAssertEqual(restored.record(first.id)?.session.context, "Updated visit context")
        XCTAssertTrue(AppointmentBrief.generate(try XCTUnwrap(restored.record(first.id)).session.trials,
            context: try XCTUnwrap(restored.record(first.id)).session.context).contains("Updated visit context"))
        XCTAssertEqual(restored.record(first.id)?.session.notes["NCT1"], "Updated first notes")
        XCTAssertTrue(try XCTUnwrap(restored.record(first.id)).session.completed)
        XCTAssertEqual(restored.session?.context, "Second visit")
    }

    func testLegacyAppointmentMigrationRetainsDataWithoutInventingDates() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        var legacy = try XCTUnwrap(AppointmentSession(trials: [try trial()], context: "Old visit"))
        legacy.notes["NCT1"] = "Keep these notes"
        legacy.completed = true
        let data = try JSONEncoder().encode(legacy)
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: path)
        let store = AppointmentStore(file: path)
        let original = try XCTUnwrap(store.records.first)
        XCTAssertNil(original.createdAt)
        XCTAssertNil(original.completedAt)
        XCTAssertEqual(try Data(contentsOf: path), data)
        XCTAssertTrue(store.start([try trial("NCT2")], context: "New visit"))
        let restored = AppointmentStore(file: path)
        XCTAssertEqual(restored.records.count, 2)
        XCTAssertEqual(restored.record(original.id)?.session.notes["NCT1"], "Keep these notes")
        XCTAssertNil(restored.record(original.id)?.createdAt)
        XCTAssertTrue(try XCTUnwrap(restored.record(original.id)).session.completed)
    }

    func testHistoryWriteFailureKeepsExistingRecords() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let store = AppointmentStore(file: path)
        XCTAssertTrue(store.start([try trial()], context: "Keep"))
        let original = try XCTUnwrap(store.records.first)
        try FileManager.default.removeItem(at: path)
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: false)
        XCTAssertFalse(store.start([try trial("NCT2")], context: "Cannot save"))
        XCTAssertEqual(store.records.map(\.id), [original.id])
        XCTAssertEqual(store.session?.context, "Keep")
        XCTAssertNotNil(store.message)
    }

    func testDeleteAppointmentsPersistsIncludingLastRecord() throws {
        let path = file(); defer { try? FileManager.default.removeItem(at: path.deletingLastPathComponent()) }
        let store = AppointmentStore(file: path)
        store.start([try trial()], context: "First")
        let first = try XCTUnwrap(store.records.first?.id)
        store.start([try trial("NCT2")], context: "Second")
        let second = try XCTUnwrap(store.records.last?.id)
        store.delete(second)
        XCTAssertEqual(store.session?.context, "First")
        XCTAssertEqual(AppointmentStore(file: path).records.map(\.id), [first])
        store.delete(first)
        let empty = AppointmentStore(file: path)
        XCTAssertTrue(empty.records.isEmpty)
        XCTAssertNil(empty.session)
        XCTAssertNil(empty.message)
        XCTAssertTrue(empty.start([try trial()], context: "New"))
    }

}
