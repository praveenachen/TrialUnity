import Foundation

/// Top-level navigation destinations for the app's single `NavigationStack`.
/// `.editStep` is a distinct case (not just `.profileStep`) because it needs
/// different Continue-button behavior: saving pops straight back to Review
/// instead of advancing to the next step in sequence.
enum AppRoute: Hashable {
    case profileStep(ProfileStep)
    case editStep(ProfileStep)
    case review
    case matching
}
