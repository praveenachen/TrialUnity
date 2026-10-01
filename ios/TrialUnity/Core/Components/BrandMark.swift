import SwiftUI

/// Shared, scalable Heart-Linked TU monogram. Color follows the app context.
struct BrandMark: View {
    var size: CGFloat = 40
    var color: Color = Theme.Color.accent

    private var safeSize: CGFloat { CGFloat(VisualNumber.dimension(Double(size))) }

    var body: some View {
        Image("BrandLogo")
            .renderingMode(.template)
            .resizable()
            .scaledToFit()
            .foregroundStyle(color)
            .frame(width: safeSize, height: safeSize)
            .accessibilityHidden(true)
    }
}

/// A compact logo + "TrialUnity" lockup for root-level screen headers.
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
