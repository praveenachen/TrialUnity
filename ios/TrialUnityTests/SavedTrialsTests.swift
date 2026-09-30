import XCTest
@testable import TrialUnityIntegration

final class SavedTrialsTests: XCTestCase {
    func result() throws -> TrialRecommendation {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "recommendations", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(TrialSearchResponse.self, from: Data(contentsOf: url)).results[0]
    }
    func testPersistenceDuplicatesAndRemoval() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("saved.json")
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        let store = SavedTrialsStore(file: file)
        let r = try result()
        store.save(r, profile: PatientProfile(draft: .sample), source: "sample-data")
        store.save(r, profile: PatientProfile(draft: .sample), source: "sample-data")
        XCTAssertEqual(store.trials.count, 1)
        let relaunched = SavedTrialsStore(file: file)
        XCTAssertEqual(relaunched.trials.first?.id, r.id)
        relaunched.remove(r.id)
        XCTAssertTrue(SavedTrialsStore(file: file).trials.isEmpty)
    }
    func testRealTrialSnapshotsWhenAvailable() throws {
        guard let path = ProcessInfo.processInfo.environment["TRIALUNITY_SAVED_LIVE_FIXTURE"] else {
            throw XCTSkip("Provide a real backend response path for the three-trial check")
        }
        struct SnapshotResponse: Decodable { let results: [TrialRecommendation]; let source: String }
        let response = try JSONDecoder().decode(SnapshotResponse.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertGreaterThanOrEqual(response.results.count, 3)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("saved.json")
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        let store = SavedTrialsStore(file: file)
        for result in response.results.prefix(3) {
            store.save(result, profile: PatientProfile(draft: .sample), source: response.source)
        }
        let restored = SavedTrialsStore(file: file).trials
        XCTAssertEqual(restored.count, 3)
        for count in 2...3 {
            let selected = Array(restored.prefix(count))
            for trial in selected { XCTAssertEqual(SavedTrialPresentation.fields(trial).count, 13) }
            let brief = AppointmentBrief.generate(selected, context: "Discussion with my care team")
            for trial in selected { XCTAssertTrue(brief.contains(trial.sourceLink)) }
        }
    }

    func testSelectionLimits() {
        var selection = TrialSelection()
        XCTAssertTrue(selection.toggle("a"))
        XCTAssertFalse(selection.canCompare)
        XCTAssertTrue(selection.toggle("b"))
        XCTAssertTrue(selection.canCompare)
        XCTAssertTrue(selection.toggle("c"))
        XCTAssertFalse(selection.toggle("d"))
        XCTAssertEqual(selection.ids.count, 3)
        selection.retain(["a"])
        XCTAssertFalse(selection.canCompare)
    }
    func testSharedBriefUsesPatientReadableRelevance() throws {
        let recommendation = try result()
        let record = SavedTrial(id: recommendation.id, title: recommendation.trial.title,
                                savedAt: nil, source: "sample-data",
                                profile: PatientProfile(draft: .sample), recommendation: recommendation)
        let brief = AppointmentBrief.generate([record], context: "Questions for my appointment")
        XCTAssertTrue(brief.contains("Why it surfaced: \(RelevanceTier(score: recommendation.score).label)"))
        XCTAssertFalse(brief.contains("Why it surfaced: \(recommendation.explanation.ranking_rationale)"))
        XCTAssertTrue(brief.contains(record.sourceLink))
        XCTAssertTrue(brief.contains("Original search condition:"))
        XCTAssertTrue(brief.contains("Questions for the care team:"))
    }
    func testMissingEvidenceAndBrief() throws {
        let partial = SavedTrial(id: "NCT00000001", title: nil, savedAt: nil, source: nil, profile: nil, recommendation: nil)
        let fields = SavedTrialPresentation.fields(partial)
        XCTAssertEqual(fields.first { $0.id == "Clinical relevance" }?.value, "Unknown")
        XCTAssertTrue(fields.first { $0.id == "ESR" }!.value.contains("Not reported"))
        XCTAssertEqual(fields.first { $0.id == "Experimental predicted representation risk" }?.value, "Not reported")
        let brief = AppointmentBrief.generate([partial], context: "")
        XCTAssertEqual(brief, AppointmentBrief.generate([partial], context: ""))
        XCTAssertTrue(brief.contains("NCT00000001"))
        XCTAssertTrue(brief.contains("https://clinicaltrials.gov/study/NCT00000001"))
        XCTAssertTrue(brief.contains("context: Not provided"))
        XCTAssertFalse(brief.contains("Patient age"))
        XCTAssertTrue(brief.contains("does not determine medical eligibility"))
        let record = SavedTrial(id: "NCT00000001", title: "Fixture", savedAt: nil, source: "sample-data", profile: nil, recommendation: try result())
        let populated = SavedTrialPresentation.fields(record)
        XCTAssertTrue(populated.first { $0.id == "ESR" }!.value.contains("Not reported"))
        XCTAssertTrue(populated.first { $0.id == "Experimental predicted representation risk" }!.value.contains("predicted"))
    }
    func testIncompleteAndCorruptRecords() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("[{\"id\":\"NCT00000001\"},{\"broken\":true}]".utf8).write(to: file)
        let store = SavedTrialsStore(file: file)
        XCTAssertEqual(store.trials.count, 1)
        XCTAssertNotNil(store.message)
    }
}
