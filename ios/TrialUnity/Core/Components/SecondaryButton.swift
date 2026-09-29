import SwiftUI

/// A quiet, text-only affordance for "Skip for now" and similar low-emphasis actions.
/// Back navigation itself uses the system back button, not this component.
struct SecondaryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Theme.Color.muted)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    SecondaryButton(title: "Skip for now") {}
        .padding()
}
