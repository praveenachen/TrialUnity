import SwiftUI

/// The one primary call-to-action style in the app (Continue / Save / "Find my matches").
struct PrimaryButton: View {
    let title: String
    var isEnabled: Bool = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    isEnabled ? Theme.Color.ink : Theme.Color.hairline,
                    in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                )
                .foregroundStyle(Theme.Color.paper)
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
