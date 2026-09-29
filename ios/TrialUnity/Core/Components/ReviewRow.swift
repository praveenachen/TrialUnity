import SwiftUI

/// One field summary on the Profile Review screen, with an inline Edit action
/// that also carries an explicit VoiceOver accessibility action of the same name.
struct ReviewRow: View {
    let label: String
    let value: String
    let onEdit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack {
                Text(label.uppercased())
                    .font(.sectionLabel)
                    .foregroundStyle(Theme.Color.muted)
                Spacer()
                Button("Edit", action: onEdit)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.Color.accent)
            }
            Text(value)
                .font(.body)
                .foregroundStyle(Theme.Color.ink)
        }
        .padding(Theme.Spacing.m)
        .background(Theme.Color.paper, in: RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous)
                .stroke(Theme.Color.hairline, lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
        .accessibilityAction(named: Text("Edit \(label)"), onEdit)
    }
}

#Preview {
    ReviewRow(label: "Condition", value: "Non-small cell lung cancer") {}
        .padding()
}
