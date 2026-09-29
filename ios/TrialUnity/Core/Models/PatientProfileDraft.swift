import Foundation
import Observation

/// In-memory draft of the patient profile being built by the onboarding wizard.
///
/// This is intentionally the *only* piece of shared state in the app right now --
/// no persistence, no networking, no backend calls. It mirrors the shape of the
/// backend's `PatientProfile` model closely enough that wiring up the real API
/// later should mean writing a mapper, not redesigning this type.
@Observable
final class PatientProfileDraft {
    var condition: String = ""
    var ageText: String = ""
    var sex: SexOption?
    var location: String = ""
    var travelPreference: TravelPreference?
    var interventionPreferences: [String] = []
    var notes: String = ""

    init() {}

    /// Parsed age, or `nil` if the text isn't a valid age between 0 and 120.
    var age: Int? {
        let trimmed = ageText.trimmingCharacters(in: .whitespaces)
        guard let value = Int(trimmed), (0...120).contains(value) else { return nil }
        return value
    }

    var isConditionValid: Bool {
        !condition.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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
