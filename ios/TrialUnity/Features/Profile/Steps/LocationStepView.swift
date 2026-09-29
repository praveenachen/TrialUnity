import SwiftUI

struct LocationStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.location.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "Where are you located?",
            subtitle: "City and state or country is enough to find nearby trial sites.",
            isOptional: true,
            continueTitle: isEditing ? "Save" : "Continue",
            onSkip: isEditing ? nil : onContinue,
            onContinue: onContinue
        ) {
            OutlinedTextField(
                placeholder: "City, state, or country",
                text: $draft.location,
                accessibilityLabelText: "Location"
            )
        }
    }
}

#Preview {
    NavigationStack {
        LocationStepView(draft: .empty, onContinue: {})
    }
}
