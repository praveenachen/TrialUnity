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
            DecisionAnchor(symbol: "mappin.and.ellipse", caption: "Where would you like to look for trial sites?")
            OutlinedTextField(
                placeholder: "City, state, or country",
                text: $draft.location,
                accessibilityLabelText: "Location"
            )
            if !draft.location.isEmpty { Text(draft.location).font(.title2.bold()) }
            VStack(alignment: .leading, spacing: 12) {
                Label("Your travel preference", systemImage: "map").font(.headline)
                Picker("Travel preference", selection: $draft.travelPreference) {
                    Text("Not specified").tag(TravelPreference?.none)
                    ForEach(TravelPreference.allCases) { option in Text(option.rawValue).tag(Optional(option)) }
                }.pickerStyle(.menu)
                Text("Saved with your profile; not used to filter trials yet.").font(.caption).foregroundStyle(Theme.Color.muted)
            }.padding(16).background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}

#Preview {
    NavigationStack {
        LocationStepView(draft: .empty, onContinue: {})
    }
}
