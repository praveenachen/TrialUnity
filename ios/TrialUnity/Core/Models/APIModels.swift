import Foundation

// Wire DTOs use the backend's snake_case names explicitly, keeping acronym mapping unambiguous.
struct PatientProfile: Codable {
    let age: Int?
    let sex: String?
    let condition: String
    let location: String?
    let intervention_preferences: [String]
    let phase_preferences: [String]
    let notes: String?

    init(draft: PatientProfileDraft) {
        func optional(_ text: String) -> String? {
            let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? nil : value
        }
        age = draft.age
        sex = draft.sex == .preferNotToSay ? nil : draft.sex?.rawValue
        condition = draft.condition.trimmingCharacters(in: .whitespacesAndNewlines)
        location = optional(draft.location)
        intervention_preferences = draft.interventionPreferences
        phase_preferences = []
        notes = optional(draft.notes)
        // Travel preference has no backend contract; it remains in the local draft.
    }
}

struct TrialSearchResponse: Codable {
    let query: String
    let total: Int
    let results: [TrialRecommendation]
    let source: String
}

struct Trial: Codable {
    let nct_id: String
    let title: String
    let status: String
    let conditions: [String]
    let interventions: [String]
    let phases: [String]
    let brief_summary: String?
    let eligibility_criteria: String?
    let sex: String?
    let minimum_age: String?
    let maximum_age: String?
    let locations: [String]
    let sponsor: String?
    let source_url: String?
    let enrollment_sex_distribution: [String: Double]?
    let enrollment_race_distribution: [String: Double]?
    let enrollment_sex_source: String?
    let enrollment_race_source: String?
    let target_enrollment: Int?
}

struct TrialRecommendation: Codable, Identifiable {
    var id: String { trial.nct_id }
    let trial: Trial
    let score: Double
    let score_breakdown: [String: Double]
    let relevance: RelevanceScores
    let explanation: MatchExplanation
    let structured_eligibility: StructuredEligibility?
    let esr: ESRResult?
    let representation_risk: RepresentationRiskPrediction?
}

struct RelevanceScores: Codable {
    let overall: Double
    let lexical: Double
    let semantic: Double
    let structured: [String: Double]
    let weighted_components: [String: Double]
    let weights: [String: Double]
}

struct MatchExplanation: Codable {
    let matched_terms: [String]
    let eligibility_notes: [String]
    let ranking_rationale: String
    let patient_friendly_summary: String
    let relevant_signals: [String]
    let manual_review_signals: [String]
}

struct StructuredEligibility: Codable {
    let status: String
    let criteria: [String: EligibilityCriterion]
}

struct EligibilityCriterion: Codable {
    let state: String
    let reason: String
}

struct ESRResult: Codable {
    let score: Double?
    let mode: String
    let evidence_coverage: Double
    let components: [String: ComponentEvidence]
    let weights_used: [String: Double]
}

struct ComponentEvidence: Codable {
    let score: Double?
    let evidence_coverage: Double
    let evidence_type: String
    let rationale: String
    let source: String?
    let missing_evidence: [String]
}

struct RepresentationRiskPrediction: Codable {
    let risk_level: String?
    let probabilities: [String: Double]?
    let confidence: Double?
    let model_version: String
    let evidence_type: String
    let drivers: [String]
    let limitations: [String]
}
