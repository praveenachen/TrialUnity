import SwiftUI

/// Splash -> Sign In -> Home. The signed-in content is keyed by the local user, so
/// switching users rebuilds all per-user state (saved trials, activity, profile draft).
struct RootView: View {
    @State private var session = UserSession()
    @State private var showsWelcome = true

    var body: some View {
        Group {
            if showsWelcome {
                WelcomeView { showsWelcome = false }
            } else if let user = session.user {
                SignedInRoot(user: user).id(user.storageKey)
            } else {
                SignInView { session.signIn(name: $0, email: $1) }
            }
        }
        .environment(session)
        .tint(Theme.Color.accent)
    }
}

private struct SignedInRoot: View {
    let user: LocalUser
    @Environment(UserSession.self) private var session
    @State private var showsMenu = false
    @State private var showsSettings = false
    @State private var showsHistory = false
    @State private var history: SearchHistoryStore
    @State private var confirmsSignOut = false
    @State private var path: [AppRoute] = []
    @State private var savedTrials: SavedTrialsStore
    @State private var appointment: AppointmentStore
    @State private var draft = PatientProfileDraft()
    @State private var recentActivity: RecentTrialActivity
    @State private var tab = 0
    @State private var appointmentPath: [UUID] = []
    @State private var activeSearch: MatchingModel?
    @State private var matchCount: Int?
    private let matchCountKey: String

    init(user: LocalUser) {
        self.user = user
        matchCountKey = "trialunity.matchCount.\(user.storageKey)"
        _matchCount = State(initialValue: UserDefaults.standard.object(forKey: matchCountKey) as? Int)
        _recentActivity = State(initialValue: RecentTrialActivity(file: RecentTrialActivity.file(forUser: user.storageKey)))
        _appointment = State(initialValue: AppointmentStore(file: AppointmentStore.file(forUser: user.storageKey)))
        _history = State(initialValue: SearchHistoryStore(file: SearchHistoryStore.file(forUser: user.storageKey)))
        _savedTrials = State(initialValue: SavedTrialsStore(file: SavedTrialsStore.file(forUser: user.storageKey)))
    }

    var body: some View {
        TabView(selection: $tab) {
            NavigationStack {
                HomeView(draft: draft, count: matchCount, explore: startNewSearch, openSaved: { tab = 2 }, openLatestSearch: openLatestSearch, openMenu: { withAnimation(.easeInOut(duration: 0.25)) { showsMenu = true } })
            }.tabItem { Label("Home", systemImage: "house") }.tag(0)
            NavigationStack(path: $path) {
                ConditionStepView(draft: draft, onContinue: { path.append(.profileStep(.age)) })
                    .navigationDestination(for: AppRoute.self, destination: destination(for:))
                    .toolbar { Button("History", systemImage: "clock.arrow.circlepath") { showsHistory = true } }
            }.tabItem { Label("Find", systemImage: "magnifyingglass") }.tag(1)
            NavigationStack { SavedTrialsView() }
                .tabItem { Label("Saved", systemImage: "bookmark") }.tag(2)
            NavigationStack(path: $appointmentPath) { AppointmentsView() }
                .tabItem { Label("Appointments", systemImage: "calendar") }.tag(3)
        }
        .overlay { sideMenu }
        .sheet(isPresented: $showsHistory) {
            NavigationStack {
                SearchHistoryView()
                    .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Done") { showsHistory = false } } }
            }
        }
        .sheet(isPresented: $showsSettings) { SettingsView() }
        .confirmationDialog("Sign out of TrialUnity?", isPresented: $confirmsSignOut, titleVisibility: .visible) {
            Button("Sign out", role: .destructive) { session.signOut() }
        } message: {
            Text("Your saved trials stay on this device and return when you sign in with the same email.")
        }
        .environment(history)
        .environment(savedTrials)
        .environment(appointment)
        .environment(\.openAppointments, {
            appointmentPath = []
            tab = 3
        })
        .environment(\.openAppointmentDetails, { id in
            appointmentPath = [id]
            tab = 3
        })
        .environment(\.recentTrialActivity, recentActivity)
        .alert("Saved trials", isPresented: Binding(get: { savedTrials.message != nil }, set: { if !$0 { savedTrials.message = nil } })) {
            Button("OK") { savedTrials.message = nil }
        } message: { Text(savedTrials.message ?? "") }
    }

    /// Home "Find trials": always a fresh search from the first step.
    private func startNewSearch() {
        draft = PatientProfileDraft()
        activeSearch = nil
        path = []
        tab = 1
    }

    /// Home "Matches": the most recent search's results (this session's, else the latest saved in history).
    private func openLatestSearch() {
        if activeSearch == nil, let record = history.records.first {
            activeSearch = MatchingModel(snapshot: record)
        }
        if activeSearch != nil { path = [.matching] }
        tab = 1
    }

    private func closeMenu(then action: (() -> Void)? = nil) {
        withAnimation(.easeInOut(duration: 0.25)) { showsMenu = false }
        action?()
    }

    @ViewBuilder private var sideMenu: some View {
        if showsMenu {
            ZStack(alignment: .leading) {
                Color.black.opacity(0.35).ignoresSafeArea()
                    .onTapGesture { closeMenu() }
                    .accessibilityLabel("Close menu").accessibilityAddTraits(.isButton)
                    .transition(.opacity)
                SideMenuView(
                    user: user,
                    openAppointments: { closeMenu { tab = 3 } },
                    openSaved: { closeMenu { tab = 2 } },
                    openHistory: { closeMenu { showsHistory = true } },
                    openSettings: { closeMenu { showsSettings = true } },
                    signOut: { closeMenu { confirmsSignOut = true } }
                )
                .transition(.move(edge: .leading))
                .gesture(DragGesture().onEnded { if $0.translation.width < -40 { closeMenu() } })
            }
        }
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
            // Fall back to a model built from the draft: on the first push the route can be
            // resolved before `activeSearch` is visible, which used to render a blank page.
            MatchingView(model: activeSearch ?? MatchingModel(profile: PatientProfile(draft: draft))) { count in
                matchCount = count
                UserDefaults.standard.set(count, forKey: matchCountKey)
            }
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
