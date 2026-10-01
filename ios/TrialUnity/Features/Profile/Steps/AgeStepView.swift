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
            continueTitle: isEditing ? "Save" : "Continue",
            isContinueEnabled: draft.isAgeValid,
            onContinue: onContinue
        ) {
            ScanSuggestionModule(draft: draft, field: .age)
            BrandedSurface {
                VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                    HStack(spacing: Theme.Spacing.m) {
                        TextField("Age", text: $draft.ageText)
                            .keyboardType(.numberPad)
                            .font(.title2.bold()).frame(maxWidth: .infinity, minHeight: 44)
                            .accessibilityLabel("Age in years")
                        Text("years").font(.subheadline).foregroundStyle(Theme.Color.muted)
                        Stepper("Adjust age", value: Binding(
                            get: { min(max(draft.age ?? 0, 0), 120) },
                            set: { draft.ageText = String($0) }
                        ), in: 0...120).labelsHidden()
                            .fixedSize().frame(minHeight: 44)
                            .accessibilityLabel("Adjust age")
                    }
                    if showsValidationMessage {
                        Text("Enter an age between 0 and 120.")
                            .font(.caption).foregroundStyle(Theme.Color.conflict)
                    }
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
