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
            VStack(spacing: 16) {
                TextField("—", text: $draft.ageText)
                    .keyboardType(.numberPad).font(.system(size: 64, weight: .bold, design: .rounded))
                    .multilineTextAlignment(.center).accessibilityLabel("Age in years")
                Text("YEARS OLD").font(.caption.bold()).foregroundStyle(Theme.Color.muted)
                Stepper("Adjust age", value: Binding(get: { draft.age ?? 0 }, set: { draft.ageText = String($0) }), in: 0...120)
                if showsValidationMessage { Text("Enter an age between 0 and 120.").foregroundStyle(Theme.Color.conflict) }
            }.padding(24).background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: 24))
            DecisionAnchor(symbol: "calendar", caption: "We compare your age with the study’s listed range. The care team confirms full eligibility.")
        }
    }
}

#Preview {
    NavigationStack {
        AgeStepView(draft: .empty, onContinue: {})
    }
}
