import SwiftUI

struct NotesStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.notes.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "Anything else worth mentioning?",
            subtitle: "Prior treatments, biomarkers, or scheduling constraints -- whatever feels relevant.",
            isOptional: true,
            continueTitle: isEditing ? "Save" : "Continue",
            onSkip: isEditing ? nil : onContinue,
            onContinue: onContinue
        ) {
            OutlinedTextField(
                placeholder: "Notes",
                text: $draft.notes,
                axis: .vertical,
                accessibilityLabelText: "Additional notes"
            )
            .frame(minHeight: 120, alignment: .top)
        }
    }
}

#Preview {
    NavigationStack {
        NotesStepView(draft: .empty, onContinue: {})
    }
}
