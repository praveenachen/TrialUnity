import SwiftUI

struct SexStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.sex.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "What is your sex?",
            continueTitle: isEditing ? "Save" : "Continue",
            isContinueEnabled: draft.isSexValid,
            onContinue: onContinue
        ) {
            VStack(spacing: Theme.Spacing.s) {
                ForEach(SexOption.allCases) { option in
                    SelectableRow(title: option.rawValue, isSelected: draft.sex == option) {
                        draft.sex = option
                    }
                }
            }
        }
    }
}

#Preview {
    NavigationStack {
        SexStepView(draft: .empty, onContinue: {})
    }
}
