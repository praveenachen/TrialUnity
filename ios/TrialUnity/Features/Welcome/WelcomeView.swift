import SwiftUI

struct WelcomeView: View {
    let onStart: () -> Void
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Text("TU").font(.system(size: 68, weight: .bold, design: .rounded))
                .accessibilityHidden(true)
            Text("TrialUnity").font(.largeTitle.bold())
            Text("Clinical trial discovery, made clearer.").font(.subheadline)
                .multilineTextAlignment(.center)
            Spacer()
            Button("Continue", action: onStart)
                .font(.headline).frame(maxWidth: .infinity, minHeight: 52)
                .foregroundStyle(Color(red: 37/255, green: 99/255, blue: 235/255))
                .background(.white, in: Capsule())
                .padding(.bottom, 24)
        }
        .padding(20).foregroundStyle(.white)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 37/255, green: 99/255, blue: 235/255).ignoresSafeArea())
    }
}

struct HomeView: View {
    @Environment(SavedTrialsStore.self) private var saved
    let draft: PatientProfileDraft
    let count: Int?
    let explore: () -> Void
    let openSaved: () -> Void
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 6) {
                    BrandHeader()
                    Text("Welcome back").font(.largeTitle.bold())
                    Text("Here’s where you left off.").foregroundStyle(Theme.Color.muted)
                }
                BrandedSurface {
                    VStack(alignment: .leading, spacing: 16) {
                        Label("YOUR SEARCH", systemImage: "magnifyingglass").font(.caption.bold())
                        Text(draft.condition.isEmpty ? "Find trials worth discussing" : draft.condition).font(.title2.bold())
                        if !draft.location.isEmpty { Text(draft.location).font(.subheadline) }
                        if let travel = draft.travelPreference { Text(travel.rawValue).font(.caption) }
                        PrimaryButton(title: count == nil ? "Find trials" : "Continue exploring", action: explore)
                    }
                }
                VStack(alignment: .leading, spacing: 14) {
                    Text("YOUR TRIAL JOURNEY").font(.caption.bold()).foregroundStyle(Theme.Color.muted)
                    GlanceGrid(items: [("Matches · current search", count.map(String.init) ?? "Not searched"), ("Saved", "\(saved.trials.count)")])
                }
                VStack(alignment: .leading, spacing: 14) {
                    Label(saved.trials.count >= 2 ? "Your shortlist is ready to compare" : "Your next step", systemImage: "bookmark")
                        .font(.headline)
                    Text(saved.trials.isEmpty ? "Start by reviewing your strongest matches, then save the trials you want to discuss." : "Review your saved trials and prepare questions for your care team.")
                        .foregroundStyle(Theme.Color.muted)
                    if let recent = saved.trials.last {
                        Text(recent.displayTitle).font(.subheadline.weight(.semibold)).lineLimit(3)
                        Text(recent.id).font(.caption)
                    }
                    Button("Open saved trials", action: openSaved).frame(minHeight: 44)
                }
            }.padding(20)
        }.background(Theme.Color.paper).navigationTitle("Home").navigationBarTitleDisplayMode(.inline)
    }
}
