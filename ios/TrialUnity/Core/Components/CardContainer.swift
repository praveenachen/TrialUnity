import SwiftUI

/// The app's one bordered-surface style: white background, a 1px hairline
/// border, and a soft corner radius -- no heavy shadow. Used for result rows,
/// saved-trial rows, and any other "card" surface, so they share one visual
/// language instead of each view inventing its own.
struct CardContainer<Content: View>: View {
    var padding: CGFloat = Theme.Spacing.l
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .background(Theme.Color.paper, in: RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                    .stroke(Theme.Color.hairline, lineWidth: 1)
            )
    }
}

/// A pale-blue branded surface -- used for the matching pipeline summary and
/// the ESR module, so both read as distinctly "TrialUnity," never confused
/// with the neutral bordered card style above or the experimental-risk module.
struct BrandedSurface<Content: View>: View {
    var padding: CGFloat = Theme.Spacing.l
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
    }
}

#Preview {
    VStack(spacing: Theme.Spacing.l) {
        CardContainer { Text("Card content") }
        BrandedSurface { Text("Branded surface content") }
    }
    .padding()
}
