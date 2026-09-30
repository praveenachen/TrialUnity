import SwiftUI

/// Native San Francisco type throughout -- no serif, no document-reader voice.
/// Hierarchy comes from size, weight, and color, not font-family changes. Every
/// style here still scales with the user's preferred Dynamic Type size. A
/// monospaced style is reserved for data provenance (NCT IDs, source labels).
///
/// Hierarchy, largest to smallest: `editorialLargeTitle` (Welcome, one screen
/// only, ~30pt bold) > `editorialTitle` (screen/section headlines, ~24pt
/// semibold) > `editorialHeadline` (item headlines that can wrap, like a trial
/// title, ~20pt semibold) > body/subheadline (system default) > `sectionLabel`
/// (small tracked caps field/section labels) > `provenance` (monospaced
/// identifiers).
extension Font {
    /// Uses relative text styles (not fixed point sizes) throughout, so every
    /// style here still scales with the user's preferred Dynamic Type size.
    static var editorialLargeTitle: Font {
        .system(.largeTitle, design: .default).weight(.bold)
    }

    static var editorialTitle: Font {
        .system(.title2, design: .default).weight(.semibold)
    }

    /// For content that can run long and wrap -- trial titles above all.
    static var editorialHeadline: Font {
        .system(.title3, design: .default).weight(.semibold)
    }

    /// Small, tracked, uppercase-styled section/field labels.
    static var sectionLabel: Font {
        .system(.caption, design: .default).weight(.semibold)
    }

    /// A large numeral for a single hero metric (the ESR score). Rounded design
    /// reads as "a number to look at," distinct from the app's prose voice.
    static var heroNumber: Font {
        .system(.largeTitle, design: .rounded).weight(.bold)
    }

    /// Reserved for NCT IDs and source/provenance text.
    static var provenance: Font {
        .system(.caption2, design: .monospaced)
    }
}
