import SwiftUI

struct TravelStepView: View {
    @Bindable var draft: PatientProfileDraft
    var isEditing: Bool = false
    let onContinue: () -> Void

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.travel.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "How far are you willing to travel?",
            isOptional: true,
            continueTitle: isEditing ? "Save" : "Continue",
            onSkip: isEditing ? nil : onContinue,
            onContinue: onContinue
        ) {
            VStack(spacing: Theme.Spacing.s) {
                ForEach(TravelPreference.allCases) { option in
                    SelectableRow(title: option.rawValue, isSelected: draft.travelPreference == option) {
                        draft.travelPreference = (draft.travelPreference == option) ? nil : option
                    }
                }
            }
        }
    }
}

#Preview {
    NavigationStack {
        TravelStepView(draft: .empty, onContinue: {})
    }
}
