import SwiftUI

/// TrialUnity's identity mark: a blue rounded-square "TU" glyph. Prominent on
/// Welcome, compact on root-level screens (Matching results) -- never repeated
/// on every detail screen, so it reads as identity, not decoration.
struct BrandMark: View {
    var size: CGFloat = 40
    var inverted = false

    private var safeSize: CGFloat { CGFloat(VisualNumber.dimension(Double(size))) }

    var body: some View {
        Text("TU")
            .font(.system(size: safeSize * 0.42, weight: .bold, design: .rounded))
            .foregroundStyle(inverted ? Color(red: 37/255, green: 99/255, blue: 235/255) : .white)
            .frame(width: safeSize, height: safeSize)
            .background(inverted ? Color.white : Theme.Color.accent, in: RoundedRectangle(cornerRadius: safeSize * 0.28, style: .continuous))
            .accessibilityHidden(true)
    }
}

/// A compact "TU  TrialUnity" lockup for root-level screen headers.
struct BrandHeader: View {
    var title: String = "TrialUnity"
    var subtitle: String?

    var body: some View {
        HStack(spacing: Theme.Spacing.s) {
            BrandMark(size: 32)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.Color.ink)
                if let subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(Theme.Color.muted)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    VStack(alignment: .leading, spacing: Theme.Spacing.l) {
        BrandMark(size: 64)
        BrandHeader(subtitle: "10 matches for Lung cancer")
    }
    .padding()
}
