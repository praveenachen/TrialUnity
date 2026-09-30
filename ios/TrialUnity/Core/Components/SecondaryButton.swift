import SwiftUI

/// A soft-blue-surface, blue-label affordance for "Skip for now" and similar
/// low-emphasis actions. Back navigation itself uses the system back button,
/// not this component.
struct SecondaryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Theme.Color.accent)
                .frame(maxWidth: .infinity, minHeight: Theme.Metrics.buttonHeight)
                .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                        .stroke(Theme.Color.accent.opacity(0.25), lineWidth: 1)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    SecondaryButton(title: "Skip for now") {}
        .padding()
}
