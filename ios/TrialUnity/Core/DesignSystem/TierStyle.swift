import SwiftUI

/// Color mapping for the presentation-only tiers in RelevanceTier.swift. Color is
/// always paired with an SF Symbol and a text label at the call site -- never
/// the sole signal -- so this stays purely decorative. Teal communicates
/// positive/compatible/observed states; amber is reserved for "needs review";
/// red is reserved for genuine conflicts, never ordinary neutral states.
extension RelevanceTier {
    var tintColor: Color {
        switch self {
        case .strong: return Theme.Color.evidence
        case .possible: return Theme.Color.attention
        case .needsReview: return Theme.Color.muted
        }
    }
}

extension EligibilitySummary {
    var tintColor: Color {
        switch self {
        case .compatible: return Theme.Color.evidence
        case .incompatible: return Theme.Color.conflict
        case .unknown: return Theme.Color.attention
        }
    }
}

extension ESREvidenceType {
    var tintColor: Color {
        switch self {
        case .observed: return Theme.Color.evidence
        case .prospective: return Theme.Color.accent
        case .insufficientData, .insufficientBenchmark: return Theme.Color.muted
        case .other: return Theme.Color.muted
        }
    }
}

extension MatchSignal.Status {
    var tintColor: Color {
        switch self {
        case .match: return Theme.Color.evidence
        case .noMatch: return Theme.Color.conflict
        case .unknown: return Theme.Color.attention
        }
    }
}
