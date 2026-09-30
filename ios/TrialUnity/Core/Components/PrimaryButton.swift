import SwiftUI

/// The one primary call-to-action style in the app (Continue / Save / "Find my matches"):
/// solid brand blue, white label.
struct PrimaryButton: View {
    let title: String
    var isEnabled: Bool = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.headline)
                .multilineTextAlignment(.center)
                .padding(.horizontal, Theme.Spacing.m)
                .padding(.vertical, Theme.Spacing.s)
                .frame(maxWidth: .infinity, minHeight: Theme.Metrics.buttonHeight)
                .background(
                    Theme.Color.accent.opacity(isEnabled ? 1 : 0.4),
                    in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                )
                .foregroundStyle(.white)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
    }
}

#Preview {
    VStack(spacing: Theme.Spacing.m) {
        PrimaryButton(title: "Continue") {}
        PrimaryButton(title: "Continue", isEnabled: false) {}
    }
    .padding()
}
