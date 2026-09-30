import SwiftUI

/// A multi-select control (compare selection, up to 3 trials) -- square check
/// glyphs distinguish it from `SelectableRow`'s single-select circles.
struct SelectionCheckRow: View {
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(
                isSelected ? "Selected for comparison" : "Select for comparison",
                systemImage: isSelected ? "checkmark.square.fill" : "square"
            )
            .font(.subheadline.weight(.medium))
            .foregroundStyle(isSelected ? Theme.Color.accent : Theme.Color.muted)
            .frame(minHeight: Theme.Metrics.minTapTarget, alignment: .leading)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}

#Preview {
    VStack(alignment: .leading, spacing: Theme.Spacing.s) {
        SelectionCheckRow(isSelected: true) {}
        SelectionCheckRow(isSelected: false) {}
    }
    .padding()
}
