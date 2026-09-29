import SwiftUI

/// A single removable tag chip (a chosen treatment/intervention preference).
struct TagChip: View {
    let title: String
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.subheadline)
            Button(action: onRemove) {
                Image(systemName: "xmark")
                    .font(.caption.weight(.semibold))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove \(title)")
        }
        .padding(.horizontal, Theme.Spacing.m)
        .padding(.vertical, Theme.Spacing.s)
        .background(Theme.Color.ink.opacity(0.06), in: Capsule())
        .foregroundStyle(Theme.Color.ink)
    }
}

/// A tappable suggestion chip (adds itself when tapped, doesn't remove).
struct SuggestionChip: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .padding(.horizontal, Theme.Spacing.m)
                .padding(.vertical, Theme.Spacing.s)
                .overlay(Capsule().stroke(Theme.Color.hairline, lineWidth: 1))
                .foregroundStyle(Theme.Color.ink)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    FlowLayout {
        TagChip(title: "Immunotherapy") {}
        TagChip(title: "Targeted therapy") {}
        SuggestionChip(title: "Chemotherapy") {}
    }
    .padding()
}
