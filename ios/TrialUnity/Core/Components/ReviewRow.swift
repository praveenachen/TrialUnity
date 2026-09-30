import SwiftUI

/// One field summary on the Profile Review screen, with an inline Edit action
/// that also carries an explicit VoiceOver accessibility action of the same name.
struct ReviewRow: View {
    let label: String
    let value: String
    let onEdit: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.m) {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                Text(label).font(.caption).foregroundStyle(Theme.Color.muted)
                Text(value).font(.subheadline).foregroundStyle(Theme.Color.ink)
                    .fixedSize(horizontal: false, vertical: true)
            }.frame(maxWidth: .infinity, alignment: .leading)
            Button("Edit", action: onEdit)
                .font(.subheadline.weight(.medium)).foregroundStyle(Theme.Color.accent)
                .frame(minWidth: 44, minHeight: 44)
        }
        .padding(.vertical, Theme.Spacing.s)
        .overlay(alignment: .bottom) { Divider() }
        .accessibilityElement(children: .combine)
        .accessibilityAction(named: Text("Edit \(label)"), onEdit)
    }
}

#Preview {
    ReviewRow(label: "Condition", value: "Non-small cell lung cancer") {}
        .padding()
}
