import Foundation

/// One row of the Match Trace: a structured signal already computed by the
/// backend (relevance.structured / structured_eligibility), relabeled for
/// display. This never recomputes a score -- it only reads the fields the
/// backend already returned and decides whether the patient provided enough
/// information to judge the signal at all.
struct MatchSignal: Identifiable, Equatable {
    enum Status: Equatable {
        case match
        case noMatch
        case unknown

        var label: String {
            switch self {
            case .match: return "Match"
            case .noMatch: return "No match"
            case .unknown: return "Needs review"
            }
        }

        var symbolName: String {
            switch self {
            case .match: return "checkmark.circle.fill"
            case .noMatch: return "exclamationmark.circle.fill"
            case .unknown: return "questionmark.circle"
            }
        }
    }

    let id: String
    let label: String
    let status: Status
    let statusText: String
    let patientValue: String
    let trialValue: String
    let sourceNote: String?
}

/// Builds the six Match Trace rows from a submitted profile and one result --
/// pure data transformation, no scoring.
enum MatchTraceBuilder {
    static func build(profile: PatientProfile, result: TrialRecommendation) -> [MatchSignal] {
        let trial = result.trial
        let structured = result.relevance.structured
        let criteria = result.structured_eligibility?.criteria ?? [:]

        return [
            conditionSignal(profile: profile, trial: trial, structured: structured),
            ageSignal(profile: profile, trial: trial, criteria: criteria),
            sexSignal(profile: profile, trial: trial, criteria: criteria),
            treatmentSignal(profile: profile, trial: trial, structured: structured),
            locationSignal(profile: profile, trial: trial, structured: structured),
            phaseSignal(profile: profile, trial: trial, structured: structured),
        ]
    }

    private static func conditionSignal(profile: PatientProfile, trial: Trial, structured: [String: Double]) -> MatchSignal {
        let matched = (structured["condition"] ?? 0) > 0
        return MatchSignal(
            id: "condition",
            label: "Condition",
            status: matched ? .match : .noMatch,
            statusText: matched ? "Match" : "No match",
            patientValue: profile.condition,
            trialValue: trial.conditions.isEmpty ? "Not listed" : trial.conditions.joined(separator: ", "),
            sourceNote: "Trial condition text, from ClinicalTrials.gov."
        )
    }

    private static func ageSignal(profile: PatientProfile, trial: Trial, criteria: [String: EligibilityCriterion]) -> MatchSignal {
        let minimum = criteria["minimum_age"]
        let maximum = criteria["maximum_age"]
        let states = [minimum?.state, maximum?.state].compactMap { $0 }
        let status: MatchSignal.Status = states.contains("incompatible") ? .noMatch
            : (states.allSatisfy { $0 == "compatible" } && !states.isEmpty) ? .match
            : .unknown
        let reasons = [minimum?.reason, maximum?.reason].compactMap { $0 }.joined(separator: " ")
        return MatchSignal(
            id: "age",
            label: "Age",
            status: status,
            statusText: status == .match ? "Compatible" : status == .noMatch ? "Incompatible" : "Needs review",
            patientValue: profile.age.map(String.init) ?? "Not provided",
            trialValue: "Min \(trial.minimum_age ?? "not specified"), max \(trial.maximum_age ?? "not specified")",
            sourceNote: reasons.isEmpty ? nil : reasons
        )
    }

    private static func sexSignal(profile: PatientProfile, trial: Trial, criteria: [String: EligibilityCriterion]) -> MatchSignal {
        let criterion = criteria["sex"]
        let status: MatchSignal.Status = criterion?.state == "compatible" ? .match
            : criterion?.state == "incompatible" ? .noMatch
            : .unknown
        return MatchSignal(
            id: "sex",
            label: "Sex",
            status: status,
            statusText: status == .match ? "Compatible" : status == .noMatch ? "Conflict found" : "Needs review",
            patientValue: profile.sex ?? "Not specified",
            trialValue: trial.sex ?? "Not specified",
            sourceNote: criterion?.reason
        )
    }

    private static func treatmentSignal(profile: PatientProfile, trial: Trial, structured: [String: Double]) -> MatchSignal {
        guard !profile.intervention_preferences.isEmpty else {
            return MatchSignal(
                id: "treatment", label: "Treatment", status: .unknown, statusText: "No preference set",
                patientValue: "Not specified",
                trialValue: trial.interventions.isEmpty ? "Not listed" : trial.interventions.joined(separator: ", "),
                sourceNote: nil
            )
        }
        let matched = (structured["intervention"] ?? 0) > 0
        return MatchSignal(
            id: "treatment",
            label: "Treatment",
            status: matched ? .match : .noMatch,
            statusText: matched ? "Match" : "Different treatment",
            patientValue: profile.intervention_preferences.joined(separator: ", "),
            trialValue: trial.interventions.isEmpty ? "Not listed" : trial.interventions.joined(separator: ", "),
            sourceNote: "Text match against trial intervention names."
        )
    }

    private static func locationSignal(profile: PatientProfile, trial: Trial, structured: [String: Double]) -> MatchSignal {
        guard let patientLocation = profile.location, !patientLocation.isEmpty else {
            return MatchSignal(
                id: "location", label: "Location", status: .unknown, statusText: "Not specified",
                patientValue: "Not specified",
                trialValue: trial.locations.isEmpty ? "Not listed" : trial.locations.joined(separator: ", "),
                sourceNote: nil
            )
        }
        let matched = (structured["location"] ?? 0) > 0
        return MatchSignal(
            id: "location",
            label: "Location",
            status: matched ? .match : .noMatch,
            statusText: matched ? "Nearby" : "Different area",
            patientValue: patientLocation,
            trialValue: trial.locations.isEmpty ? "Not listed" : trial.locations.joined(separator: ", "),
            sourceNote: "Text match against listed trial sites; not a distance calculation."
        )
    }

    private static func phaseSignal(profile: PatientProfile, trial: Trial, structured: [String: Double]) -> MatchSignal {
        guard !profile.phase_preferences.isEmpty else {
            return MatchSignal(
                id: "phase", label: "Phase", status: .unknown, statusText: "No preference",
                patientValue: "No preference",
                trialValue: trial.phases.isEmpty ? "Not listed" : trial.phases.joined(separator: ", "),
                sourceNote: nil
            )
        }
        let matched = (structured["phase"] ?? 0) > 0
        return MatchSignal(
            id: "phase",
            label: "Phase",
            status: matched ? .match : .noMatch,
            statusText: matched ? "Preferred" : "Different phase",
            patientValue: profile.phase_preferences.joined(separator: ", "),
            trialValue: trial.phases.isEmpty ? "Not listed" : trial.phases.joined(separator: ", "),
            sourceNote: nil
        )
    }
}
