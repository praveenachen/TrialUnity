import Foundation

/// Patient sex, as captured for structured eligibility matching. "Prefer not to say"
/// is a first-class, equally valid answer -- not a placeholder for a missing one.
enum SexOption: String, CaseIterable, Identifiable {
    case female = "Female"
    case male = "Male"
    case preferNotToSay = "Prefer not to say"

    var id: String { rawValue }
}

/// How far the patient is willing to travel for a trial site.
enum TravelPreference: String, CaseIterable, Identifiable {
    case local = "Local sites only"
    case regional = "Within a few hours"
    case nationwide = "Anywhere in my country"
    case remoteFriendly = "Open to remote or decentralized trials"

    var id: String { rawValue }
}
