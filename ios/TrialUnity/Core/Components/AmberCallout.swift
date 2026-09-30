import SwiftUI

/// A single pale-amber callout grouping "needs review"-type notes -- used
/// instead of a list of tiny raw bullet Labels, so manual-review items read as
/// one clear block rather than scattered fine print.
struct AmberCallout: View {
    let title: String
    let items: [String]
    var symbolName: String = "exclamationmark.circle.fill"

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                Label(title, systemImage: symbolName)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.Color.attention)
                VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                    ForEach(items, id: \.self) { item in
                        Text("•  \(item)")
                            .font(.caption)
                            .foregroundStyle(Theme.Color.ink)
                    }
                }
            }
            .padding(Theme.Spacing.m)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.Color.attention.opacity(0.1), in: RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
            .accessibilityElement(children: .combine)
        }
    }
}

#Preview {
    AmberCallout(title: "Needs manual review", items: [
        "Full eligibility criteria and site availability require study-team review.",
        "Patient age is missing.",
    ])
    .padding()
}
