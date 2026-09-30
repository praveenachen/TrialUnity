import SwiftUI

/// A slim, labeled bar making "how much evidence backs this score" visually
/// obvious rather than buried in a percentage nobody reads.
struct CoverageBar: View {
    let coverage: Double
    var label: String = "Evidence coverage"

    // Guards NaN/±infinity from the source, not just at the point of use: a
    // non-finite `coverage` (should never happen from decoded JSON, but this is
    // the boundary that protects every downstream CGFloat/frame/progress use
    // below) is treated as 0, then clamped to a valid percentage range.
    private var clamped: Double {
        guard coverage.isFinite else { return 0 }
        return min(max(coverage, 0), 1)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack {
                Text(label)
                    .font(.sectionLabel)
                    .foregroundStyle(Theme.Color.muted)
                Spacer()
                Text(clamped, format: .percent.precision(.fractionLength(0)))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.Color.ink)
            }
            GeometryReader { proxy in
                // During navigation transitions SwiftUI can propose an unconstrained
                // (infinite) width before layout settles. `.infinity * 0` (a real,
                // common `clamped` value for zero evidence coverage) is NaN, which
                // CoreGraphics then rejects with a runtime warning -- so the width
                // is clamped to a finite, non-negative value before use, every time.
                let safeWidth = CGFloat(VisualNumber.dimension(Double(proxy.size.width)))
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.Color.hairline)
                    Capsule()
                        .fill(Theme.Color.accent)
                        .frame(width: safeWidth * clamped)
                }
            }
            .frame(height: 6)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label): \(Int((clamped * 100).rounded())) percent")
    }
}

#Preview {
    VStack(spacing: Theme.Spacing.l) {
        CoverageBar(coverage: 0.6)
        CoverageBar(coverage: 0.0)
        CoverageBar(coverage: 1.0)
    }
    .padding()
}
