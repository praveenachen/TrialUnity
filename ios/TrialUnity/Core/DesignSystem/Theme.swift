import SwiftUI

/// TrialUnity's visual tokens: a restrained, editorial "clinical research" palette
/// and spacing scale, not a generic SaaS design system. Colors are defined as
/// asset-catalog color sets (see Assets.xcassets) so light/dark mode and future
/// contrast tuning stay out of the view code entirely.
enum Theme {
    enum Color {
        /// Warm, paper-like background -- not stark white.
        static let paper = SwiftUI.Color("PaperBackground")
        /// Primary text.
        static let ink = SwiftUI.Color("InkText")
        /// Secondary/supporting text.
        static let muted = SwiftUI.Color("MutedText")
        /// Hairline borders and dividers.
        static let hairline = SwiftUI.Color("Hairline")
        /// A single, restrained accent used sparingly for selection and links.
        static let accent = SwiftUI.Color("ClinicalAccent")
    }

    enum Spacing {
        static let xs: CGFloat = 4
        static let s: CGFloat = 8
        static let m: CGFloat = 16
        static let l: CGFloat = 24
        static let xl: CGFloat = 40
    }

    enum Radius {
        static let field: CGFloat = 10
        static let control: CGFloat = 14
    }
}
