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
    private static let guidedSteps: [ProfileStep] = [.condition, .age, .sex, .location, .treatment, .notes]

    var stepNumber: Int { (Self.guidedSteps.firstIndex(of: self) ?? 3) + 1 }

    static var totalSteps: Int { guidedSteps.count }

    /// The next step in sequence, or `nil` if this is the last one (the caller
    /// should advance to the Review screen instead).
    var next: ProfileStep? {
        guard let index = Self.guidedSteps.firstIndex(of: self), index + 1 < Self.guidedSteps.count else { return nil }
        return Self.guidedSteps[index + 1]
    }
}
