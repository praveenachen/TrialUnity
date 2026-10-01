import SwiftUI

/// A section of the Trial Passport: an editorial heading, an optional subtitle,
/// and content -- separated by a hairline rule, not a filled card. Keeps the
/// passport reading like a document with sections, not a stack of dashboard tiles.
struct PassportSection<Content: View>: View {
    let title: String
    var subtitle: String?
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
            Divider().overlay(Theme.Color.hairline)
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                Text(title.uppercased())
                    .font(.sectionLabel)
                    .foregroundStyle(Theme.Color.muted)
                if let subtitle {
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundStyle(Theme.Color.muted)
                }
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
    }
}

#Preview {
    VerticalScrollView {
        PassportSection(title: "Your match") {
            Text("Match content goes here.")
        }
        .padding()
    }
}
