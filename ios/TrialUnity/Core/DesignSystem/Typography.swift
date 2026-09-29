import SwiftUI

/// Editorial type treatment layered on top of Dynamic Type text styles, so every
/// font here still scales with the user's preferred text size. Headlines use a
/// serif design for a document/research feel; a monospaced style is reserved for
/// data provenance (NCT IDs, source labels) introduced in a later phase.
extension Font {
    static var editorialLargeTitle: Font {
        .system(.largeTitle, design: .serif).weight(.semibold)
    }

    static var editorialTitle: Font {
        .system(.title2, design: .serif).weight(.semibold)
    }

    /// Small, all-caps-styled section/field labels.
    static var sectionLabel: Font {
        .system(.caption, design: .default).weight(.semibold)
    }

    /// Reserved for NCT IDs and source/provenance text in later phases.
    static var provenance: Font {
        .system(.caption2, design: .monospaced)
    }
}
