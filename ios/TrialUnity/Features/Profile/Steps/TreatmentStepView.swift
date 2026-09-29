import SwiftUI

struct TreatmentStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    @State private var newPreference: String = ""

    private let suggestions = ["Immunotherapy", "Chemotherapy", "Targeted therapy", "Surgery", "Radiation therapy"]

    private var availableSuggestions: [String] {
        suggestions.filter { suggestion in
            !draft.interventionPreferences.contains { $0.caseInsensitiveCompare(suggestion) == .orderedSame }
        }
    }

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.treatment.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "Any treatments you're especially interested in?",
            subtitle: "Add as many as apply -- this narrows, but never excludes, your matches.",
            isOptional: true,
            continueTitle: isEditing ? "Save" : "Continue",
            onSkip: isEditing ? nil : onContinue,
            onContinue: onContinue
        ) {
            VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                OutlinedTextField(
                    placeholder: "Type a treatment and press return",
                    text: $newPreference,
                    accessibilityLabelText: "Add a treatment preference"
                )
                .onSubmit(addTypedPreference)

                if !draft.interventionPreferences.isEmpty {
                    FlowLayout(spacing: Theme.Spacing.s) {
                        ForEach(draft.interventionPreferences, id: \.self) { preference in
                            TagChip(title: preference) {
                                draft.removeInterventionPreference(preference)
                            }
                        }
                    }
                }

                if !availableSuggestions.isEmpty {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        Text("Suggestions")
                            .font(.sectionLabel)
                            .foregroundStyle(Theme.Color.muted)
                        FlowLayout(spacing: Theme.Spacing.s) {
                            ForEach(availableSuggestions, id: \.self) { suggestion in
                                SuggestionChip(title: suggestion) {
                                    draft.addInterventionPreference(suggestion)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func addTypedPreference() {
        draft.addInterventionPreference(newPreference)
        newPreference = ""
    }
}

#Preview {
    NavigationStack {
        TreatmentStepView(draft: .empty, onContinue: {})
    }
}
