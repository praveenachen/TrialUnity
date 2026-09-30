import Foundation

/// Presentation only: never changes backend scores or eligibility.
enum PatientPresentation {
    static func evidence(_ coverage: Double) -> String {
        guard coverage.isFinite, coverage > 0 else { return "Limited evidence" }
        if coverage >= 0.8 { return "Broad evidence" }
        return coverage >= 0.4 ? "Moderate evidence" : "Limited evidence"
    }
    static func summary(_ text: String?) -> String {
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "Study summary not reported." }
        // An excerpt of the original text, not a generated medical interpretation.
        let sentences = text.components(separatedBy: ". ").prefix(2).joined(separator: ". ")
        return sentences.count > 320 ? String(sentences.prefix(320)) + "…" : sentences
    }
    static func confirmations(_ result: TrialRecommendation) -> [String] {
        var items: [String] = []
        if result.trial.maximum_age == nil { items.append("Maximum age · not listed in the record") }
        if result.trial.minimum_age == nil { items.append("Minimum age · not listed in the record") }
        if result.structured_eligibility?.status == "incompatible" { items.append("A profile detail conflicts with the listed criteria") }
        items.append("Full eligibility · ask the study team to review")
        items.append("Site availability · confirm before arranging travel")
        return items
    }
    static func location(_ trial: Trial, near location: String?) -> String? {
        guard let location, !location.isEmpty else { return trial.locations.first }
        return trial.locations.first { $0.localizedCaseInsensitiveContains(location) } ?? trial.locations.first
    }
}

/// Bounds for custom layout arithmetic. Normal unbounded SwiftUI proposals are
/// resolved before they reach a concrete size, coordinate or CoreGraphics path.
enum VisualNumber {
    static func dimension(_ value: Double) -> Double {
        value.isFinite ? min(max(0, value), 1_000_000) : 0
    }
    static func fraction(_ numerator: Double, _ denominator: Double) -> Double {
        guard numerator.isFinite, denominator.isFinite, denominator > 0 else { return 0 }
        return min(max(numerator / denominator, 0), 1)
    }
}
