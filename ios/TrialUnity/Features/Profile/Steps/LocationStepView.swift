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
            VStack(alignment: .leading, spacing: 12) {
                Label("Your travel preference", systemImage: "map").font(.headline)
                Picker("Travel preference", selection: $draft.travelPreference) {
                    Text("Not specified").tag(TravelPreference?.none)
                    ForEach(TravelPreference.allCases) { option in Text(option.rawValue).tag(Optional(option)) }
                }.pickerStyle(.menu)
                Text("Saved with your profile; not used to filter trials yet.").font(.caption).foregroundStyle(Theme.Color.muted)
            }.frame(maxWidth: .infinity, alignment: .leading)
                .padding(Theme.Metrics.cardPadding)
                .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.card))
        }
    }
}

#Preview {
    NavigationStack {
        LocationStepView(draft: .empty, onContinue: {})
    }
}
