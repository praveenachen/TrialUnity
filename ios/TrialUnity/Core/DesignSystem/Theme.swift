import SwiftUI

/// TrialUnity's visual tokens: a clean, modern healthcare palette built on white,
/// TrialUnity blue, and a secondary evidence teal -- not a generic SaaS or
/// document-reader look. Colors are defined as asset-catalog color sets (see
/// Assets.xcassets) so light/dark mode and future contrast tuning stay out of
/// the view code entirely.
///
/// Blue is the primary brand/action color (CTAs, links, primary emphasis).
/// Teal is secondary and communicates evidence, success, or compatibility --
/// it never becomes the app's dominant color. Semantic colors are always
/// paired with an SF Symbol and a text label at the call site (see
/// RelevanceTier/EligibilitySummary/ESREvidenceType in Core/Models) -- color
/// alone never carries a state.
enum Theme {
    enum Color {
        /// Screen background -- clean white (near-black in dark mode).
        static let paper = SwiftUI.Color("PaperBackground")
        /// A soft, pale-blue surface for branded modules (matching pipeline,
        /// ESR) that need to read as "TrialUnity" without being a full card.
        static let surface = SwiftUI.Color("SurfaceBlue")
        /// Primary text.
        static let ink = SwiftUI.Color("InkText")
        /// Secondary/supporting text -- a slate gray, not a tinted color.
        static let muted = SwiftUI.Color("MutedText")
        /// Hairline borders and dividers.
        static let hairline = SwiftUI.Color("Hairline")
        /// The primary brand/action color: CTAs, links, selection, primary
        /// interactive emphasis. TrialUnity blue.
        static let accent = SwiftUI.Color("ClinicalAccent")
        /// Secondary brand color: positive/observed evidence, compatibility,
        /// strong relevance. Used to communicate "good," never as a CTA color.
        static let evidence = SwiftUI.Color("EvidenceTeal")
        /// Needs review / unknown / prospective (not-yet-observed) evidence.
        static let attention = SwiftUI.Color("AttentionColor")
        /// Reserved for genuinely destructive actions (e.g. removing a saved
        /// trial) -- never used for ordinary navigation or informational state.
        static let conflict = SwiftUI.Color("ConflictColor")
        /// Reserved exclusively for the experimental Representation Risk
        /// prediction -- a neutral slate so it never reads as equal in
        /// authority to the observed-evidence ESR module.
        static let experimental = SwiftUI.Color("ExperimentalColor")
        /// Appointment-prep accent (Home journey tile only).
        static let violet = SwiftUI.Color(red: 124/255, green: 92/255, blue: 214/255)
    }

    enum Spacing {
        static let xs: CGFloat = 4
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 16
        static let xl: CGFloat = 24
        static let xxl: CGFloat = 32
    }

    enum Radius {
        static let field: CGFloat = 14
        static let control: CGFloat = 14
        /// Bordered card surfaces (result rows, saved rows, branded modules).
        static let card: CGFloat = 16
    }

    enum Metrics {
        /// Apple's minimum recommended hit target; anything tappable should be
        /// at least this tall (via `.frame(minHeight:)`), even when its visible
        /// content is smaller.
        static let minTapTarget: CGFloat = 44
        /// Standard screen horizontal padding.
        static let screenPadding: CGFloat = 20
        static let cardPadding: CGFloat = 16
        static let compactPadding: CGFloat = 12
        static let sectionSpacing: CGFloat = 24
        static let buttonHeight: CGFloat = 50
        static let actionGutter: CGFloat = 28
    }
}
