import Foundation

/// Presentation-only qualitative bucketing of the backend's already-computed
/// relevance score. This never recomputes relevance -- it only labels a number
/// the backend already produced, for a UI that shouldn't show a bare percentage
/// as if it were a probability of eligibility.
enum RelevanceTier: String, CaseIterable {
    case strong
    case possible
    case needsReview

    /// Thresholds are presentation-only and not fit to any dataset -- they exist
    /// purely to turn a continuous score into three explainable labels.
    init(score: Double) {
        switch score {
        case 0.6...: self = .strong
        case 0.3..<0.6: self = .possible
        default: self = .needsReview
        }
    }

    var label: String {
        switch self {
        case .strong: return "Strong relevance"
        case .possible: return "Possible relevance"
        case .needsReview: return "Needs review"
        }
    }

    var symbolName: String {
        switch self {
        case .strong: return "checkmark.circle.fill"
        case .possible: return "circle.lefthalf.filled"
        case .needsReview: return "questionmark.circle"
        }
    }
}

/// A short, patient-facing read of `StructuredEligibility.status`. Never claims
/// eligibility is confirmed -- "compatible" here means "no known disqualifier."
enum EligibilitySummary {
    case compatible
    case incompatible
    case unknown

    init(status: String?) {
        switch status {
        case "compatible": self = .compatible
        case "incompatible": self = .incompatible
        default: self = .unknown
        }
    }

    var label: String {
        switch self {
        case .compatible: return "No known eligibility conflicts"
        case .incompatible: return "Eligibility conflict found"
        case .unknown: return "Eligibility needs review"
        }
    }

    var shortLabel: String {
        switch self {
        case .compatible: return "Looks compatible"
        case .incompatible: return "Conflict found"
        case .unknown: return "Needs review"
        }
    }

    var symbolName: String {
        switch self {
        case .compatible: return "checkmark.circle.fill"
        case .incompatible: return "exclamationmark.circle.fill"
        case .unknown: return "questionmark.circle"
        }
    }
}

/// A single evidence label used across ESR components and eligibility criteria to
/// keep "we don't know" visually and verbally distinct from a real 0.
enum EvidenceState {
    case known(String)
    case unknown

    init(score: Double?, formatted: (Double) -> String) {
        if let score { self = .known(formatted(score)) } else { self = .unknown }
    }

    var displayValue: String {
        switch self {
        case .known(let value): return value
        case .unknown: return "—"
        }
    }

    var isUnknown: Bool {
        if case .unknown = self { return true }
        return false
    }
}

enum ESREvidenceType {
    case observed
    case prospective
    case insufficientData
    case insufficientBenchmark
    case other(String)

    init(rawValue: String) {
        switch rawValue {
        case "observed_enrollment", "observed_geographic", "observed_distribution_comparison":
            self = .observed
        case "protocol_inclusivity":
            self = .prospective
        case "insufficient_data":
            self = .insufficientData
        case "insufficient_benchmark":
            self = .insufficientBenchmark
        default:
            self = .other(rawValue)
        }
    }

    var label: String {
        switch self {
        case .observed: return "Observed evidence"
        case .prospective: return "Prospective evidence"
        case .insufficientData: return "Data unavailable"
        case .insufficientBenchmark: return "Benchmark unavailable"
        case .other(let raw): return raw
        }
    }

    var symbolName: String {
        switch self {
        case .observed: return "checkmark.seal"
        case .prospective: return "clock.arrow.circlepath"
        case .insufficientData, .insufficientBenchmark: return "minus.circle"
        case .other: return "info.circle"
        }
    }
}

enum RiskLevelDisplay {
    static func label(_ rawValue: String?) -> String {
        guard let rawValue else { return "Unavailable" }
        return rawValue.capitalized
    }

    /// Extracts the leading feature name (the raw backend key) from a driver
    /// string such as "num_sites (importance 0.552)". Never mutates or discards
    /// the raw key -- callers that need it for anything other than display
    /// (logging, analytics, etc.) should use this, not `driverLabel`.
    static func shortDriver(_ raw: String) -> String {
        guard let range = raw.range(of: " (") else { return raw }
        return String(raw[..<range.lowerBound])
    }

    /// Known raw feature keys mapped to patient-friendly labels. Anything not
    /// listed here falls back to a humanized version of the raw key (underscores
    /// to spaces, capitalized) rather than showing a raw snake_case identifier.
    private static let driverLabels: [String: String] = [
        "num_sites": "Number of trial sites",
        "num_regions": "Geographic reach",
        "target_enrollment_log": "Target enrollment size",
    ]

    /// A patient-friendly display label for a backend driver string. The raw key
    /// (see `shortDriver`) is only ever used to look one up -- never displayed
    /// directly unless it has no known friendly mapping.
    static func driverLabel(_ raw: String) -> String {
        let key = shortDriver(raw)
        if let label = driverLabels[key] { return label }
        return key.replacingOccurrences(of: "_", with: " ").capitalized
    }
}
