import SwiftUI

struct RootView: View {
    @State private var path: [AppRoute] = []
    @State private var draft = PatientProfileDraft()

    var body: some View {
        NavigationStack(path: $path) {
            WelcomeView(onStart: { path.append(.profileStep(.condition)) })
                .navigationDestination(for: AppRoute.self, destination: destination(for:))
        }
        .tint(Theme.Color.accent)
    }

    @ViewBuilder
    private func destination(for route: AppRoute) -> some View {
        switch route {
        case .profileStep(let step):
            stepView(for: step, isEditing: false)
        case .editStep(let step):
            stepView(for: step, isEditing: true)
        case .review:
            ProfileReviewView(
                draft: draft,
                onEdit: { step in path.append(.editStep(step)) },
                onContinue: { path.append(.matching) }
            )
        case .matching:
            MatchingView(profile: PatientProfile(draft: draft))
        }
    }

    /// Builds the view for a single wizard step, wiring its Continue button to
    /// either advance to the next step (normal flow) or pop back to Review
    /// (editing an already-reviewed answer).
    @ViewBuilder
    private func stepView(for step: ProfileStep, isEditing: Bool) -> some View {
        let onContinue = {
            if isEditing {
                path.removeLast()
            } else if let next = step.next {
                path.append(.profileStep(next))
            } else {
                path.append(.review)
            }
        }

        switch step {
        case .condition:
            ConditionStepView(draft: draft, isEditing: isEditing, onContinue: onContinue)
        case .age:
            AgeStepView(draft: draft, isEditing: isEditing, onContinue: onContinue)
        case .sex:
            SexStepView(draft: draft, isEditing: isEditing, onContinue: onContinue)
        case .location:
            LocationStepView(draft: draft, isEditing: isEditing, onContinue: onContinue)
        case .travel:
            TravelStepView(draft: draft, isEditing: isEditing, onContinue: onContinue)
        case .treatment:
            TreatmentStepView(draft: draft, isEditing: isEditing, onContinue: onContinue)
        case .notes:
            NotesStepView(draft: draft, isEditing: isEditing, onContinue: onContinue)
        }
    }
}

#Preview {
    RootView()
}
