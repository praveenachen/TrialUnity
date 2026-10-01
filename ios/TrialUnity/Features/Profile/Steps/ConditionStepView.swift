import SwiftUI

struct ConditionStepView: View {
    @Bindable var draft: PatientProfileDraft
    @State private var showsScanner = false
    var isEditing: Bool = false
    let onContinue: () -> Void

    var body: some View {
        WizardScaffold(
            stepNumber: ProfileStep.condition.stepNumber,
            totalSteps: ProfileStep.totalSteps,
            title: "What condition are you exploring trials for?",
            continueTitle: isEditing ? "Save" : "Continue",
            isContinueEnabled: draft.isConditionValid,
            onContinue: onContinue
        ) {
            ScanSuggestionModule(draft: draft, field: .condition)
            OutlinedTextField(
                placeholder: "e.g. Non-small cell lung cancer",
                text: $draft.condition,
                axis: .vertical,
                accessibilityLabelText: "Condition or diagnosis"
            )
            Button("Scan medical document", systemImage: "doc.viewfinder") { showsScanner = true }
                .frame(minHeight: 44)
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                FlowLayout {
                    ForEach(["Lung cancer", "Breast cancer", "Type 2 diabetes"], id: \.self) { example in
                        SuggestionChip(title: example) { draft.condition = example }
                    }
                }
            }
        }
        .sheet(isPresented: $showsScanner) { DocumentScannerView(draft: draft) }
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
