import Foundation
import Observation

/// A backend snapshot plus the profile used to obtain it. No search response or funnel is retained.
struct SavedTrial: Codable, Identifiable {
    let id: String
    let title: String?
    let savedAt: Date?
    let source: String?
    let profile: PatientProfile?
    let recommendation: TrialRecommendation?
    var displayTitle: String { title?.isEmpty == false ? title! : id }
    var sourceLink: String { "https://clinicaltrials.gov/study/\(id)" }
}

@Observable final class SavedTrialsStore {
    private(set) var trials: [SavedTrial] = []
    var message: String?
    private let file: URL

    init(file: URL = URL.applicationSupportDirectory.appendingPathComponent("TrialUnity/saved-trials-v1.json")) {
        self.file = file
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        do {
            // Decode records independently so one obsolete record doesn't hide all saved trials.
            let data = try Data(contentsOf: file)
            guard let rows = try JSONSerialization.jsonObject(with: data) as? [Any] else {
                throw CocoaError(.fileReadCorruptFile)
            }
            for row in rows {
                guard let data = try? JSONSerialization.data(withJSONObject: row),
                      let record = try? JSONDecoder().decode(SavedTrial.self, from: data),
                      !record.id.isEmpty else { message = "Some saved trials could not be restored. Save them again from results."; continue }
                if !contains(record.id) { trials.append(record) }
            }
        } catch { message = "Saved trials could not be read. Your saved file has not been changed." }
    }

    func contains(_ id: String) -> Bool { trials.contains { $0.id == id } }

    func save(_ result: TrialRecommendation, profile: PatientProfile, source: String) {
        guard !contains(result.id) else { return }
        persist(trials + [SavedTrial(id: result.id, title: result.trial.title, savedAt: Date(), source: source, profile: profile, recommendation: result)])
    }

    func remove(_ id: String) { persist(trials.filter { $0.id != id }) }

    private func persist(_ updated: [SavedTrial]) {
        do {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(updated)
            try data.write(to: file, options: .atomic)
            #if os(iOS)
            try? FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: file.path)
            #endif
            trials = updated
            message = nil
        } catch { message = "Couldn't save this change. Please try again." }
    }
}

struct TrialSelection {
    private(set) var ids: Set<String> = []
    mutating func toggle(_ id: String) -> Bool {
        if ids.remove(id) != nil { return true }
        guard ids.count < 3 else { return false }
        ids.insert(id)
        return true
    }
    mutating func retain(_ available: Set<String>) { ids.formIntersection(available) }
    var canCompare: Bool { (2...3).contains(ids.count) }
}

struct ComparisonField: Identifiable {
    let id: String
    let value: String
}

enum SavedTrialPresentation {
    static func number(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(2))) } ?? "Not reported"
    }
    /// Humanizes a raw backend snake_case token (e.g. "insufficient_data") for
    /// display when there's no dedicated friendly-label type for it here.
    private static func humanized(_ raw: String) -> String {
        raw.replacingOccurrences(of: "_", with: " ").capitalized
    }

    static func fields(_ record: SavedTrial) -> [ComparisonField] {
        let r = record.recommendation
        func list(_ values: [String]?) -> String { values?.isEmpty == false ? values!.joined(separator: ", ") : "Not reported" }
        func component(_ key: String) -> String {
            guard let c = r?.esr?.components[key] else { return "Not reported" }
            let evidence = ESREvidenceType(rawValue: c.evidence_type).label
            return "\(number(c.score)) · \(evidence) · \(c.rationale)"
        }
        return [
            .init(id: "Clinical relevance", value: r.map { RelevanceTier(score: $0.score).label } ?? "Unknown"),
            .init(id: "Structured eligibility", value: EligibilitySummary(status: r?.structured_eligibility?.status).label),
            .init(id: "Phase", value: list(r?.trial.phases)),
            .init(id: "Interventions", value: list(r?.trial.interventions)),
            .init(id: "Locations", value: list(r?.trial.locations)),
            .init(id: "Age range", value: "Minimum: \(r?.trial.minimum_age ?? "Not reported"); maximum: \(r?.trial.maximum_age ?? "Not reported")"),
            .init(id: "Sex requirement", value: r?.trial.sex ?? "Not reported"),
            .init(id: "ESR", value: "\(number(r?.esr?.score)) · evidence: \(r?.esr.map { humanized($0.mode) } ?? "Unknown")"),
            .init(id: "ESR evidence coverage", value: r?.esr.map { $0.evidence_coverage.formatted(.percent) } ?? "Not reported"),
            .init(id: "Socioeconomic access", value: component("socioeconomic")),
            .init(id: "Sex representation / inclusivity", value: component("sex")),
            .init(id: "Race representation", value: component("race")),
            .init(id: "Experimental predicted representation risk", value: r?.representation_risk.map { "\(RiskLevelDisplay.label($0.risk_level)) · predicted, not observed ESR evidence. \($0.limitations.joined(separator: " "))" } ?? "Not reported")
        ]
    }
}

enum AppointmentBrief {
    /// Friendly labels for structured eligibility criterion keys, so the brief
    /// reads as prose rather than raw backend field names like "minimum_age".
    private static let criterionLabels: [String: String] = [
        "minimum_age": "Minimum age", "maximum_age": "Maximum age",
        "sex": "Sex", "recruitment_status": "Recruitment status",
    ]
    private static func criterionLabel(_ key: String) -> String {
        criterionLabels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized
    }

    static func generate(_ trials: [SavedTrial], context: String) -> String {
        var lines = ["TrialUnity — appointment shortlist", "Patient-entered condition/context: \(context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Not provided" : context)"]
        for item in trials.prefix(3) {
            let r = item.recommendation
            lines += ["", "\(item.displayTitle) (\(item.id))", "Source: \(item.source ?? "Unknown")", "Registry: \(item.sourceLink)", "Saved snapshot; confirm current recruitment and site details.", "Why it surfaced: \(r?.explanation.ranking_rationale ?? "Not reported")"]
            if let profile = item.profile { lines.append("Original search condition: \(profile.condition)") }
            lines.append("Eligibility items to confirm:")
            if let eligibility = r?.structured_eligibility {
                for key in eligibility.criteria.keys.sorted() {
                    if let c = eligibility.criteria[key] {
                        let label = criterionLabel(key)
                        lines.append("- \(label): \(EligibilitySummary(status: c.state).shortLabel). \(c.reason)")
                        if c.state != "compatible" {
                            lines.append("- Ask the care team to clarify \(label): \(c.reason)")
                        }
                    }
                }
            } else { lines.append("- Structured eligibility: Unknown") }
            for note in r?.explanation.eligibility_notes ?? [] { lines.append("- \(note)") }
            lines.append("Unknown / manual review:")
            for note in r?.explanation.manual_review_signals ?? [] { lines.append("- \(note)") }
            lines += ["- Full eligibility and current site availability require care-team confirmation.", "Questions for the care team:", "- Does my age meet this study's full eligibility criteria?", "- Would my previous treatment history affect eligibility?", "- Is this site currently accepting participants?", "- Which additional eligibility criteria and tests need review?"]
        }
        lines += ["", "TrialUnity supports trial navigation and does not determine medical eligibility."]
        return lines.joined(separator: "\n")
    }
}

extension SavedTrial {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        title = try? values.decode(String.self, forKey: .title)
        savedAt = try? values.decode(Date.self, forKey: .savedAt)
        source = try? values.decode(String.self, forKey: .source)
        profile = try? values.decode(PatientProfile.self, forKey: .profile)
        recommendation = try? values.decode(TrialRecommendation.self, forKey: .recommendation)
    }
}
