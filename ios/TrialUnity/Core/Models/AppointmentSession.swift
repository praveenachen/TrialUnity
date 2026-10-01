import Foundation
import Observation

struct AppointmentItem: Codable, Identifiable {
    let id: String
    let text: String
    let isQuestion: Bool
}

extension AppointmentBrief {
    static let disclaimer = "TrialUnity supports trial navigation and does not determine medical eligibility."

    static func questions(for trial: SavedTrial) -> [String] {
        var items = ["Would my previous treatment history affect eligibility?", "Is additional testing required?"]
        if let location = trial.recommendation?.trial.locations.first {
            items.insert("Is the \(location) site currently recruiting?", at: 0)
        }
        return items
    }

    static func discussionItems(for trial: SavedTrial) -> [AppointmentItem] {
        var confirmations = trial.recommendation.map(PatientPresentation.confirmations) ?? ["Full eligibility and site availability"]
        if let r = trial.recommendation {
            for key in (r.structured_eligibility?.criteria.keys.sorted() ?? []) {
                if let criterion = r.structured_eligibility?.criteria[key] {
                    confirmations.append("\(key.replacingOccurrences(of: "_", with: " ").capitalized): \(criterion.reason)")
                }
            }
            confirmations += r.explanation.eligibility_notes + r.explanation.manual_review_signals
        }
        var seen = Set<String>()
        let unique = confirmations.filter { !$0.isEmpty && seen.insert($0).inserted }
        return unique.enumerated().map { AppointmentItem(id: "confirm-\($0.offset)", text: $0.element, isQuestion: false) }
            + questions(for: trial).enumerated().map { AppointmentItem(id: "question-\($0.offset)", text: $0.element, isQuestion: true) }
    }

    static func summary(_ session: AppointmentSession) -> String {
        var lines = ["Appointment summary", "Checked means Discussed only; it does not mean eligible, confirmed, or resolved."]
        for trial in session.trials {
            lines += ["", "\(trial.displayTitle) (\(trial.id))", "Discussed items:"]
            let items = discussionItems(for: trial)
            let discussed = items.filter { session.isDiscussed($0.id, trialID: trial.id) }
            lines += discussed.isEmpty ? ["None"] : discussed.map { "- \($0.text)" }
            lines.append("Open items:")
            let open = items.filter { !session.isDiscussed($0.id, trialID: trial.id) }
            lines += open.isEmpty ? ["None"] : open.map { "- \($0.text)" }
            lines += ["User notes:", session.notes[trial.id]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false ? session.notes[trial.id]! : "None"]
        }
        // Retain the existing evidence, registry links, original context, and disclaimer.
        lines += ["", generate(session.trials, context: session.context)]
        return lines.joined(separator: "\n")
    }
}

struct AppointmentSession: Codable {
    let trials: [SavedTrial]
    var context: String
    var discussed: [String: Set<String>] = [:]
    var notes: [String: String] = [:]
    var currentIndex = 0
    var reviewedIDs: Set<String> = []
    var completed = false

    init?(trials: [SavedTrial], context: String) {
        var seen = Set<String>()
        let unique = trials.filter { seen.insert($0.id).inserted }
        guard (1...3).contains(unique.count) else { return nil }
        self.trials = unique
        self.context = context
    }
    var selectedIDs: [String] { trials.map(\.id) }
    var isValid: Bool {
        (1...3).contains(trials.count) && Set(selectedIDs).count == trials.count && trials.indices.contains(currentIndex)
    }
    func isDiscussed(_ itemID: String, trialID: String) -> Bool { discussed[trialID]?.contains(itemID) == true }
    var discussedQuestionCount: Int {
        trials.reduce(0) { total, trial in
            total + AppointmentBrief.discussionItems(for: trial).filter { $0.isQuestion && isDiscussed($0.id, trialID: trial.id) }.count
        }
    }
    var remainingQuestionCount: Int {
        trials.reduce(0) { $0 + AppointmentBrief.questions(for: $1).count } - discussedQuestionCount
    }
    /// One note entry per trial; lines/words are not counted as separate notes.
    var noteCount: Int { trials.filter { !(notes[$0.id] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.count }
}

struct AppointmentRecord: Codable, Identifiable {
    let id: UUID
    let createdAt: Date?
    var updatedAt: Date?
    var completedAt: Date?
    var session: AppointmentSession
}

private struct AppointmentArchive: Codable {
    var currentID: UUID?
    var records: [AppointmentRecord]
}

@Observable final class AppointmentStore {
    private(set) var records: [AppointmentRecord] = []
    private var currentID: UUID?
    var session: AppointmentSession? { record()?.session }
    func record(_ id: UUID? = nil) -> AppointmentRecord? {
        records.first { $0.id == (id ?? currentID) }
    }
    var message: String?
    private let file: URL
    static func file(forUser key: String) -> URL {
        URL.applicationSupportDirectory.appendingPathComponent("TrialUnity/appointment-\(key).json")
    }
    init(file: URL) {
        self.file = file
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        do {
            let data = try Data(contentsOf: file)
            if let archive = try? JSONDecoder().decode(AppointmentArchive.self, from: data) {
                guard Set(archive.records.map(\.id)).count == archive.records.count,
                      archive.records.allSatisfy({ $0.session.isValid }),
                      (archive.records.isEmpty ? archive.currentID == nil : archive.records.contains(where: { $0.id == archive.currentID })) else { throw CocoaError(.fileReadCorruptFile) }
                records = archive.records
                currentID = archive.currentID
            } else {
                let restored = try JSONDecoder().decode(AppointmentSession.self, from: data)
                guard restored.isValid else { throw CocoaError(.fileReadCorruptFile) }
                // The old format never recorded dates; do not invent an appointment date.
                let legacy = AppointmentRecord(id: UUID(), createdAt: nil, updatedAt: nil, completedAt: nil, session: restored)
                records = [legacy]
                currentID = legacy.id
            }
        } catch { message = "The appointment could not be restored. The saved file has not been changed." }
    }
    @discardableResult func start(_ trials: [SavedTrial], context: String) -> Bool {
        guard let value = AppointmentSession(trials: trials, context: context) else {
            message = "Select 1–3 saved trials."; return false
        }
        let record = AppointmentRecord(id: UUID(), createdAt: Date(), updatedAt: Date(), completedAt: nil, session: value)
        return persist(records + [record], currentID: record.id)
    }
    func delete(_ id: UUID) {
        guard records.contains(where: { $0.id == id }) else { return }
        let remaining = records.filter { $0.id != id }
        _ = persist(remaining, currentID: currentID == id ? remaining.last?.id : currentID)
    }
    func toggle(_ item: AppointmentItem, trialID: String, recordID: UUID? = nil) {
        update(recordID) { value in
            guard let trial = value.trials.first(where: { $0.id == trialID }),
                  AppointmentBrief.discussionItems(for: trial).contains(where: { $0.id == item.id }) else { return }
            var items = value.discussed[trialID] ?? []
            if !items.insert(item.id).inserted { items.remove(item.id) }
            value.discussed[trialID] = items
        }
    }
    func setContext(_ context: String, recordID: UUID? = nil) {
        update(recordID) { $0.context = context }
    }
    func setNote(_ note: String, trialID: String, recordID: UUID? = nil) {
        update(recordID) { if $0.selectedIDs.contains(trialID) { $0.notes[trialID] = note } }
    }
    func move(to index: Int, recordID: UUID? = nil) {
        update(recordID) { if $0.trials.indices.contains(index) { $0.currentIndex = index } }
    }
    func reviewCurrent(recordID: UUID? = nil) { update(recordID) { $0.reviewedIDs.insert($0.trials[$0.currentIndex].id) } }
    func complete(recordID: UUID? = nil) {
        update(recordID) {
            $0.reviewedIDs.insert($0.trials[$0.currentIndex].id)
            if Set($0.selectedIDs).isSubset(of: $0.reviewedIDs) { $0.completed = true }
        }
    }
    func reopen(recordID: UUID? = nil) { update(recordID) { $0.completed = false } }
    private func update(_ id: UUID?, _ body: (inout AppointmentSession) -> Void) {
        guard let index = records.firstIndex(where: { $0.id == (id ?? currentID) }), let currentID else { return }
        var values = records
        let wasCompleted = values[index].session.completed
        body(&values[index].session)
        values[index].updatedAt = Date()
        if values[index].session.completed && !wasCompleted { values[index].completedAt = Date() }
        if !values[index].session.completed { values[index].completedAt = nil }
        _ = persist(values, currentID: currentID)
    }
    @discardableResult private func persist(_ values: [AppointmentRecord], currentID: UUID?) -> Bool {
        do {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(AppointmentArchive(currentID: currentID, records: values)).write(to: file, options: .atomic)
            #if os(iOS)
            try? FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: file.path)
            #endif
            records = values; self.currentID = currentID; message = nil; return true
        } catch { message = "Couldn't save appointment changes. Please try again."; return false }
    }
}
