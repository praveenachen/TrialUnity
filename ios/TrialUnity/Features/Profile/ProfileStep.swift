import Foundation

/// The ordered steps of the guided patient-profile wizard.
enum ProfileStep: Int, CaseIterable, Hashable {
    case condition
    case age
    case sex
    case location
    case travel
    case treatment
    case notes

    /// 1-based position, for the "Step X of Y" progress indicator.
    var stepNumber: Int { rawValue + 1 }

    static var totalSteps: Int { allCases.count }

    /// The next step in sequence, or `nil` if this is the last one (the caller
    /// should advance to the Review screen instead).
    var next: ProfileStep? {
        ProfileStep(rawValue: rawValue + 1)
    }
}
