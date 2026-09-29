import SwiftUI

/// A hairline-bordered text field matching the app's restrained, document-like
/// form styling, used in place of the default UITextField chrome.
struct OutlinedTextField: View {
    let placeholder: String
    @Binding var text: String
    var axis: Axis = .horizontal
    var keyboardType: UIKeyboardType = .default
    var accessibilityLabelText: String?

    var body: some View {
        TextField(placeholder, text: $text, axis: axis)
            .keyboardType(keyboardType)
            .textFieldStyle(.plain)
            .padding(Theme.Spacing.m)
            .background(Theme.Color.paper, in: RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous)
                    .stroke(Theme.Color.hairline, lineWidth: 1)
            )
            .accessibilityLabel(accessibilityLabelText ?? placeholder)
    }
}

#Preview {
    OutlinedTextField(placeholder: "e.g. Non-small cell lung cancer", text: .constant(""))
        .padding()
}
