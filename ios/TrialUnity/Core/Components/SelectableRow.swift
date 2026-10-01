import SwiftUI

/// A single-select list row (sex, travel preference, etc.) with a large tap target
/// and a selection state that reads clearly to VoiceOver.
struct SelectableRow: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Text(title)
                    .font(.body)
                    .foregroundStyle(Theme.Color.ink)
                Spacer()
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? Theme.Color.accent : Theme.Color.hairline)
                    .imageScale(.large)
            }
            .padding(Theme.Spacing.m)
            .frame(maxWidth: .infinity, minHeight: Theme.Metrics.buttonHeight)
            .background(isSelected ? Theme.Color.surface : Theme.Color.paper, in: RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous)
                    .stroke(isSelected ? Theme.Color.accent : Theme.Color.hairline, lineWidth: isSelected ? 1.5 : 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.pressable)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}

#Preview {
    VStack(spacing: Theme.Spacing.s) {
        SelectableRow(title: "Female", isSelected: true) {}
        SelectableRow(title: "Male", isSelected: false) {}
    }
    .padding()
}
