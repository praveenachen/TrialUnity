import SwiftUI

struct ProfileReviewView: View {
    @Bindable var draft: PatientProfileDraft
    let onEdit: (ProfileStep) -> Void
    let onContinue: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.l) {
                VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                    Text("Review your profile")
                        .font(.editorialTitle)
                        .foregroundStyle(Theme.Color.ink)
                    Text("Double-check these details before we look for matching trials.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.Color.muted)
                }

                VStack(spacing: Theme.Spacing.s) {
                    ReviewRow(
                        label: "Condition",
                        value: draft.condition.isEmpty ? "Not provided" : draft.condition,
                        onEdit: { onEdit(.condition) }
                    )
                    ReviewRow(
                        label: "Age",
                        value: draft.age.map { "\($0)" } ?? "Not provided",
                        onEdit: { onEdit(.age) }
                    )
                    ReviewRow(
                        label: "Sex",
                        value: draft.sex?.rawValue ?? "Not provided",
                        onEdit: { onEdit(.sex) }
                    )
                    ReviewRow(
                        label: "Location",
                        value: draft.location.isEmpty ? "Not provided" : draft.location,
                        onEdit: { onEdit(.location) }
                    )
                    ReviewRow(
                        label: "Travel preference",
                        value: draft.travelPreference?.rawValue ?? "Not specified",
                        onEdit: { onEdit(.travel) }
                    )
                    ReviewRow(
                        label: "Treatment preferences",
                        value: draft.interventionPreferences.isEmpty
                            ? "None specified"
                            : draft.interventionPreferences.joined(separator: ", "),
                        onEdit: { onEdit(.treatment) }
                    )
                    ReviewRow(
                        label: "Notes",
                        value: draft.notes.isEmpty ? "None" : draft.notes,
                        onEdit: { onEdit(.notes) }
                    )
                }
            }
            .padding(Theme.Spacing.l)
        }
        .background(Theme.Color.paper)
        .safeAreaInset(edge: .bottom) {
            PrimaryButton(title: "Find matching trials", action: onContinue)
                .padding(.horizontal, Theme.Spacing.l)
                .padding(.top, Theme.Spacing.s)
                .padding(.bottom, Theme.Spacing.m)
                .background(.bar)
        }
        .navigationTitle("Review")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack {
        ProfileReviewView(draft: .sample, onEdit: { _ in }, onContinue: {})
    }
}
