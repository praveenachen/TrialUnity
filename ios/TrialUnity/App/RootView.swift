import SwiftUI

struct RootView: View {
    @State private var path: [AppRoute] = []
    @State private var savedTrials = SavedTrialsStore()
    @State private var draft = PatientProfileDraft()

    @AppStorage("hasEnteredTrialUnity") private var hasEntered = false
    @State private var tab = 0
    @State private var activeSearch: MatchingModel?
    @State private var matchCount: Int?

    var body: some View {
        Group {
            if !hasEntered && savedTrials.trials.isEmpty {
                WelcomeView { hasEntered = true; tab = 1 }
            } else {
                TabView(selection: $tab) {
                    NavigationStack {
                        HomeView(draft: draft, count: matchCount, explore: { if activeSearch != nil { path = [.matching] }; tab = 1 }, openSaved: { tab = 2 })
                    }.tabItem { Label("Home", systemImage: "house") }.tag(0)
                    NavigationStack(path: $path) {
                        ConditionStepView(draft: draft, onContinue: { path.append(.profileStep(.age)) })
                            .navigationDestination(for: AppRoute.self, destination: destination(for:))
                    }.tabItem { Label("Find", systemImage: "magnifyingglass") }.tag(1)
                    NavigationStack { SavedTrialsView() }
                        .tabItem { Label("Saved", systemImage: "bookmark") }.tag(2)
                }
            }
        }
        .environment(savedTrials)
        .alert("Saved trials", isPresented: Binding(get: { savedTrials.message != nil }, set: { if !$0 { savedTrials.message = nil } })) {
            Button("OK") { savedTrials.message = nil }
        } message: { Text(savedTrials.message ?? "") }
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
                onContinue: { activeSearch = MatchingModel(profile: PatientProfile(draft: draft)); matchCount = nil; path.append(.matching) }
            )
        case .matching:
            if let activeSearch { MatchingView(model: activeSearch) { matchCount = $0 } }
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
