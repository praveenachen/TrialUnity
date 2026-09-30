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
                .frame(maxWidth: .infinity, minHeight: 50)
                .background(
                    isEnabled ? Theme.Color.accent : Theme.Color.hairline,
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
