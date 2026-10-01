import Foundation
import Observation

struct SearchHistoryRecord: Codable, Identifiable {
    let id: UUID
    let date: Date
    let profile: PatientProfile
    let response: TrialSearchResponse
}

@Observable final class SearchHistoryStore {
    private(set) var records: [SearchHistoryRecord] = []
    var message: String?
    private let file: URL
    static func file(forUser key: String) -> URL {
        URL.applicationSupportDirectory.appendingPathComponent("TrialUnity/search-history-\(key).json")
    }
    init(file: URL) {
        self.file = file
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        do { records = try JSONDecoder().decode([SearchHistoryRecord].self, from: Data(contentsOf: file)) }
        catch { message = "Search history could not be restored." }
    }
    func record(id: UUID, profile: PatientProfile, response: TrialSearchResponse) {
        guard !records.contains(where: { $0.id == id }) else { return }
        let updated = [SearchHistoryRecord(id: id, date: Date(), profile: profile, response: response)] + records
        do {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(updated).write(to: file, options: .atomic)
            #if os(iOS)
            try? FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: file.path)
            #endif
            records = updated; message = nil
        } catch { message = "This search could not be saved to history." }
    }
}
