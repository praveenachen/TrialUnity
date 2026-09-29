import SwiftUI

struct AgeStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    private var showsValidationMessage: Bool {
        !draft.ageText.isEmpty && !draft.isAgeValid
    }

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.age.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "How old are you?",
            subtitle: "Age helps us check eligibility criteria before matching.",
            continueTitle: isEditing ? "Save" : "Continue",
            isContinueEnabled: draft.isAgeValid,
            onContinue: onContinue
        ) {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                OutlinedTextField(
                    placeholder: "Age",
                    text: $draft.ageText,
                    keyboardType: .numberPad,
                    accessibilityLabelText: "Age in years"
                )
                .frame(maxWidth: 140)

                if showsValidationMessage {
                    Text("Enter an age between 0 and 120.")
                        .font(.caption)
                        .foregroundStyle(.red)
                        .accessibilityLabel("Enter an age between 0 and 120")
                }
            }
        }
    }
}

#Preview {
    NavigationStack {
        AgeStepView(draft: .empty, onContinue: {})
    }
}
