import SwiftUI

struct TreatmentStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    @State private var newPreference: String = ""
    @State private var choosingTreatments = false

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
            title: "Which treatments interest you?",
            isOptional: true,
            continueTitle: isEditing ? "Save" : "Continue",
            onSkip: isEditing ? nil : onContinue,
            onContinue: onContinue
        ) {
            ScanSuggestionModule(draft: draft, field: .treatment)
            VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                SelectableRow(title: "No preference", isSelected: draft.interventionPreferences.isEmpty && !choosingTreatments) {
                    draft.interventionPreferences = []
                    newPreference = ""
                    choosingTreatments = false
                }
                if !choosingTreatments && draft.interventionPreferences.isEmpty {
                    Button("Choose treatments") { choosingTreatments = true }
                        .font(.subheadline.weight(.semibold)).frame(minHeight: 44)
                }
                if choosingTreatments || !draft.interventionPreferences.isEmpty {
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
