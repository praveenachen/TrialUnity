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
        .padding(.vertical, Theme.Spacing.xs)
        .background(tint.opacity(0.1), in: Capsule())
    }
}

/// Trial recruitment status as plain colored text -- no capsule, border or
/// background -- so it sits quietly inside metadata lines like
/// "NCT01234567 · Recruiting · Phase 2". Recruiting is brand blue; other states are muted.
struct TrialStatusText: View {
    let status: String

    private var normalized: String { status.uppercased().replacingOccurrences(of: " ", with: "_") }
    private var tint: Color {
        switch normalized {
        case "RECRUITING", "ENROLLING_BY_INVITATION": return Theme.Color.accent
        case "NOT_YET_RECRUITING": return Theme.Color.attention
        default: return Theme.Color.muted
        }
    }

    var body: some View {
        Text(status.replacingOccurrences(of: "_", with: " ").capitalized)
            .font(.caption.weight(.semibold))
            .foregroundStyle(tint)
    }
}

#Preview {
    VStack(alignment: .leading, spacing: Theme.Spacing.s) {
        StatusPill(text: "Strong relevance", symbolName: "checkmark.circle.fill", tint: Theme.Color.accent)
        StatusPill(text: "Needs review", symbolName: "questionmark.circle", tint: Theme.Color.muted)
    }
    .padding()
}
