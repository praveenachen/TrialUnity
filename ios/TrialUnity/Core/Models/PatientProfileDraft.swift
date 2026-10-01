import Foundation
import Observation

/// In-memory profile draft. Travel preferences remain local to the app.
@Observable
final class PatientProfileDraft {
    var condition: String = ""
    var ageText: String = ""
    var sex: SexOption?
    var location: String = ""
    var travelPreference: TravelPreference?
    var interventionPreferences: [String] = []
    var notes: String = ""
    // Shares the existing in-memory draft lifetime across profile navigation.
    var scanSuggestions: [ScanCandidate] = []
    var acceptedScanSuggestionIDs: Set<String> = []

    func keepScanSuggestions(_ candidates: [ScanCandidate]) {
        let accepted = scanSuggestions.filter { acceptedScanSuggestionIDs.contains($0.id) }
        scanSuggestions = accepted + candidates.filter { !acceptedScanSuggestionIDs.contains($0.id) }
    }

    @discardableResult func useScanSuggestion(_ id: String, replaceExisting: Bool = false) -> Bool {
        guard let item = scanSuggestions.first(where: { $0.id == id }),
              !acceptedScanSuggestionIDs.contains(id) else { return false }
        if !replaceExisting && !DocumentProfileAssist.conflicts([item], draft: self).isEmpty { return false }
        switch item.field {
        case .condition: condition = item.value
        case .age: ageText = item.value
        case .treatment: addInterventionPreference(item.value)
        case .biomarker:
            let value = "Biomarkers: \(item.value)"
            if !notes.contains(value) { notes += (notes.isEmpty ? "" : "\n") + value }
        }
        acceptedScanSuggestionIDs.insert(id)
        return true
    }

    init() {}

    /// Parsed age, or `nil` if the text isn't a valid age between 0 and 120.
    var age: Int? {
        let trimmed = ageText.trimmingCharacters(in: .whitespaces)
        guard let value = Int(trimmed), (0...120).contains(value) else { return nil }
        return value
    }

    var isConditionValid: Bool {
        condition.trimmingCharacters(in: .whitespacesAndNewlines).count >= 2
    }

    var isAgeValid: Bool { age != nil }

    var isSexValid: Bool { sex != nil }

    func addInterventionPreference(_ raw: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard !interventionPreferences.contains(where: { $0.caseInsensitiveCompare(trimmed) == .orderedSame }) else { return }
        interventionPreferences.append(trimmed)
    }

    func removeInterventionPreference(_ value: String) {
        interventionPreferences.removeAll { $0 == value }
    }
}

extension PatientProfileDraft {
    /// Sample data for SwiftUI previews -- never used at runtime.
    static var sample: PatientProfileDraft {
        let draft = PatientProfileDraft()
        draft.condition = "Non-small cell lung cancer"
        draft.ageText = "58"
        draft.sex = .female
        draft.location = "Denver, Colorado"
        draft.travelPreference = .regional
        draft.interventionPreferences = ["Immunotherapy", "Targeted therapy"]
        draft.notes = "Diagnosed six months ago, currently between treatment cycles."
        return draft
    }

    /// A fresh, empty draft -- for previewing the start of the wizard.
    static var empty: PatientProfileDraft { PatientProfileDraft() }
}
