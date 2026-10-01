import SwiftUI

struct NotesStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.notes.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "Any additional context?",
            isOptional: true,
            continueTitle: isEditing ? "Save" : "Continue",
            onSkip: isEditing ? nil : onContinue,
            onContinue: onContinue
        ) {
            ScanSuggestionModule(draft: draft, field: .biomarker)
            FlowLayout(spacing: Theme.Spacing.s) {
                ForEach(["Biomarkers", "Prior treatment", "Travel limits", "Scheduling"], id: \.self) { topic in
                    SuggestionChip(title: topic) {
                        let prompt = "\(topic): "
                        if !draft.notes.contains(prompt) {
                            draft.notes += (draft.notes.isEmpty ? "" : "\n") + prompt
                        }
                    }
                }
            }
            OutlinedTextField(
                placeholder: "Notes",
                text: $draft.notes,
                axis: .vertical,
                accessibilityLabelText: "Additional notes"
            )
        }
    }
}

#Preview {
    NavigationStack {
        NotesStepView(draft: .empty, onContinue: {})
    }
}
