import SwiftUI

struct ConditionStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.condition.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "What condition are you exploring trials for?",
            subtitle: "Use the term your doctor uses, or describe it in your own words.",
            continueTitle: isEditing ? "Save" : "Continue",
            isContinueEnabled: draft.isConditionValid,
            onContinue: onContinue
        ) {
            OutlinedTextField(
                placeholder: "e.g. Non-small cell lung cancer",
                text: $draft.condition,
                axis: .vertical,
                accessibilityLabelText: "Condition or diagnosis"
            )
        }
    }
}

#Preview {
    NavigationStack {
        ConditionStepView(draft: .empty, onContinue: {})
    }
}

#Preview("Filled") {
    NavigationStack {
        ConditionStepView(draft: .sample, onContinue: {})
    }
}
