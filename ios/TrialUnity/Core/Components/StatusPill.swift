import SwiftUI

/// A compact symbol + text indicator. Color is always secondary to the symbol and
/// text -- never the only way a state is communicated, so this reads fine without
/// color vision and to VoiceOver (which announces the label, not the tint).
struct StatusPill: View {
    let text: String
    let symbolName: String
    var tint: Color = Theme.Color.ink

    var body: some View {
        Label {
            Text(text)
                .font(.caption.weight(.medium))
        } icon: {
            Image(systemName: symbolName)
                .font(.caption)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, Theme.Spacing.s)
        .padding(.vertical, 6)
        .background(tint.opacity(0.1), in: Capsule())
    }
}

#Preview {
    VStack(alignment: .leading, spacing: Theme.Spacing.s) {
        StatusPill(text: "Strong relevance", symbolName: "checkmark.circle.fill", tint: Theme.Color.accent)
        StatusPill(text: "Needs review", symbolName: "questionmark.circle", tint: Theme.Color.muted)
    }
    .padding()
}
