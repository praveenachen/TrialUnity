import SwiftUI

/// Color mapping for the presentation-only tiers in RelevanceTier.swift. Color is
/// always paired with an SF Symbol and a text label elsewhere -- never the sole
/// signal -- so this stays purely decorative.
extension RelevanceTier {
    var tintColor: Color {
        switch self {
        case .strong: return Theme.Color.accent
        case .possible: return Theme.Color.muted
        case .needsReview: return Theme.Color.ink
        }
    }
}

extension EligibilitySummary {
    var tintColor: Color {
        switch self {
        case .compatible: return Theme.Color.accent
        case .incompatible: return .red
        case .unknown: return Theme.Color.muted
        }
    }
}

extension ESREvidenceType {
    var tintColor: Color {
        switch self {
        case .observed: return Theme.Color.accent
        case .prospective: return Theme.Color.muted
        case .insufficientData, .insufficientBenchmark: return Theme.Color.hairline
        case .other: return Theme.Color.muted
        }
    }
}
