import SwiftUI
import CryptoKit

struct WelcomeView: View {
    let onStart: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var logo = false
    @State private var wordmark = false
    @State private var button = false

    private let splashBlue = Color(red: 0, green: 68/255, blue: 245/255)

    var body: some View {
        GeometryReader { geometry in
            let compact = geometry.size.height < 650
            let markSize: CGFloat = compact ? 132 : 148

            VStack(spacing: 0) {
                VStack(spacing: 16) {
                    BrandMark(size: markSize, color: .white)
                        .opacity(logo ? 1 : 0)
                        .scaleEffect(logo ? 1 : 0.94)
                    Text("TrialUnity")
                        .font(.system(size: compact ? 42 : 46, weight: .semibold))
                        .foregroundStyle(.white)
                        .minimumScaleFactor(0.75)
                        .lineLimit(1)
                        .opacity(wordmark ? 1 : 0)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                Button(action: onStart) {
                    Text("Continue")
                        .font(.system(size: 21, weight: .semibold))
                        .frame(maxWidth: .infinity, minHeight: 56)
                        .foregroundStyle(splashBlue)
                        .background(.white, in: Capsule())
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .padding(.bottom, compact ? 30 : 40)
                .opacity(button ? 1 : 0)
            }
            .padding(.horizontal, 32)
            .background {
                RadialGradient(
                    colors: [Color(red: 0, green: 112/255, blue: 1), splashBlue],
                    center: UnitPoint(x: 0.5, y: 0.43),
                    startRadius: 0,
                    endRadius: geometry.size.height * 0.48
                )
                .ignoresSafeArea()
            }
        }
        .environment(\.colorScheme, .dark)
        .onAppear(perform: play)
    }

    private func play() {
        guard !reduceMotion else { logo = true; wordmark = true; button = true; return }
        withAnimation(.easeOut(duration: 0.45)) { logo = true }
        withAnimation(.easeOut(duration: 0.35).delay(0.15)) { wordmark = true }
        withAnimation(.easeOut(duration: 0.3).delay(0.3)) { button = true }
    }
}

/// The locally stored "account": a name and email kept on this device. There is no
/// server or authentication; it only scopes local data (saved trials, activity) per person.
struct LocalUser: Codable, Equatable {
    let name: String
    let email: String

    var firstName: String { name.split(separator: " ").first.map(String.init) ?? name }

    /// Stable per-email key so signing back in restores the same local data.
    var storageKey: String {
        let digest = SHA256.hash(data: Data(email.lowercased().utf8))
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}

@Observable final class UserSession {
    private static let key = "trialunity.localUser"
    private let defaults: UserDefaults
    private(set) var user: LocalUser?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        user = defaults.data(forKey: Self.key).flatMap { try? JSONDecoder().decode(LocalUser.self, from: $0) }
    }

    func signIn(name: String, email: String) {
        let user = LocalUser(name: name.trimmingCharacters(in: .whitespacesAndNewlines),
                             email: email.trimmingCharacters(in: .whitespacesAndNewlines))
        defaults.set(try? JSONEncoder().encode(user), forKey: Self.key)
        self.user = user
    }

    func signOut() {
        defaults.removeObject(forKey: Self.key)
        user = nil
    }
}

struct SignInView: View {
    let onSignIn: (String, String) -> Void
    @State private var name = ""
    @State private var email = ""
    @FocusState private var focus: Field?
    private enum Field { case name, email }

    private var isValid: Bool {
        let e = email.trimmingCharacters(in: .whitespaces)
        return !name.trimmingCharacters(in: .whitespaces).isEmpty
            && e.contains("@") && e.split(separator: "@").last?.contains(".") == true
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                    BrandMark(size: 56)
                    Text("Sign in").font(.largeTitle.bold()).padding(.top, Theme.Spacing.m)
                    Text("Tell us who you are so your saved trials and activity stay with you on this device.")
                        .foregroundStyle(Theme.Color.muted)
                }
                VStack(alignment: .leading, spacing: Theme.Spacing.l) {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        Text("Name").font(.subheadline.weight(.semibold))
                        OutlinedTextField(placeholder: "Your name", text: $name)
                            .textContentType(.name).submitLabel(.next)
                            .focused($focus, equals: .name)
                            .onSubmit { focus = .email }
                    }
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        Text("Email").font(.subheadline.weight(.semibold))
                        OutlinedTextField(placeholder: "you@example.com", text: $email, keyboardType: .emailAddress)
                            .textContentType(.emailAddress).textInputAutocapitalization(.never).autocorrectionDisabled()
                            .submitLabel(.go)
                            .focused($focus, equals: .email)
                            .onSubmit(submit)
                    }
                }
                PrimaryButton(title: "Continue", isEnabled: isValid, action: submit)
                Text("No password or account is created. This stays on your device.")
                    .font(.caption).foregroundStyle(Theme.Color.muted)
            }
            .padding(Theme.Metrics.screenPadding)
        }
        .scrollDismissesKeyboard(.interactively)
        .foregroundStyle(Theme.Color.ink)
        .background(Theme.Color.paper.ignoresSafeArea())
    }

    private func submit() { if isValid { onSignIn(name, email) } }
}

/// Recently viewed trials and generated briefs, persisted per user so Home isn't
/// blank on every launch. Saved snapshots and their persistence stay separate.
@Observable final class RecentTrialActivity {
    private(set) var trials: [SavedTrial] = []
    private(set) var briefIDs: Set<UUID> = []
    private let file: URL?

    private struct Stored: Codable { var trials: [SavedTrial]; var briefIDs: [UUID] }

    static func file(forUser key: String) -> URL {
        URL.applicationSupportDirectory.appendingPathComponent("TrialUnity/recent-\(key).json")
    }

    init(file: URL? = nil) {
        self.file = file
        guard let file, let data = try? Data(contentsOf: file),
              let stored = try? JSONDecoder().decode(Stored.self, from: data) else { return }
        trials = stored.trials
        briefIDs = Set(stored.briefIDs)
    }

    func recordBrief(_ id: UUID) { briefIDs.insert(id); persist() }

    func record(_ result: TrialRecommendation, profile: PatientProfile, source: String) {
        trials.removeAll { $0.id == result.id }
        trials.insert(SavedTrial(id: result.id, title: result.trial.title, savedAt: nil,
                                 source: source, profile: profile, recommendation: result), at: 0)
        trials = Array(trials.prefix(20))
        persist()
    }

    private func persist() {
        guard let file else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? JSONEncoder().encode(Stored(trials: trials, briefIDs: Array(briefIDs))).write(to: file, options: .atomic)
    }
}

private struct RecentTrialActivityKey: EnvironmentKey {
    static let defaultValue: RecentTrialActivity? = nil
}

extension EnvironmentValues {
    var recentTrialActivity: RecentTrialActivity? {
        get { self[RecentTrialActivityKey.self] }
        set { self[RecentTrialActivityKey.self] = newValue }
    }
}

struct HomeView: View {
    @Environment(UserSession.self) private var session
    @Environment(SavedTrialsStore.self) private var saved
    @Environment(\.recentTrialActivity) private var activity
    @State private var openedRecent: SavedTrial?
    @State private var confirmsSignOut = false
    let draft: PatientProfileDraft
    let count: Int?
    let explore: () -> Void
    let openSaved: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                Button(action: explore) {
                    HStack(spacing: 12) {
                        Image(systemName: "magnifyingglass")
                        Text("Find trials").font(.headline)
                        Spacer(minLength: 8)
                        Image(systemName: "arrow.right")
                    }
                    .padding(.horizontal, 16).frame(minHeight: 48)
                    .foregroundStyle(Theme.Color.accent)
                    .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: 14))
                }.buttonStyle(.plain)

                VStack(alignment: .leading, spacing: 14) {
                    Text("Your trial journey")
                        .font(.title3.bold())
                        .foregroundStyle(Theme.Color.ink)
                    HStack(alignment: .top, spacing: 10) {
                        journeyValue(count.map(String.init) ?? "—", label: "Matches",
                                     icon: "magnifyingglass", detail: "Matching trials\nready",
                                     colors: [.pink, .purple, .blue])
                        journeyValue(String(saved.trials.count), label: "Saved",
                                     icon: "bookmark", detail: "Bookmarked\nfor later",
                                     colors: [.cyan, .blue, .purple])
                        journeyValue(String(activity?.briefIDs.count ?? 0), label: "Briefs",
                                     icon: "doc.text", detail: "Summary ready\nto review",
                                     colors: [.pink, .purple, .cyan])
                    }
                }

                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Recent activity").font(.title3.bold())
                        Spacer()
                        if let trials = activity?.trials, !trials.isEmpty {
                            NavigationLink("See all") { RecentActivityView() }
                                .font(.subheadline).foregroundStyle(Theme.Color.accent).frame(minHeight: 44)
                        }
                    }
                    if let trials = activity?.trials, !trials.isEmpty {
                        VStack(spacing: 0) {
                            ForEach(Array(trials.prefix(2).enumerated()), id: \.element.id) { index, record in
                                if index > 0 { Divider() }
                                RecentActivityRow(record: record) { openedRecent = record }
                            }
                        }
                        .padding(.horizontal, 16)
                        .background(Theme.Color.paper, in: RoundedRectangle(cornerRadius: Theme.Radius.card))
                        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card).stroke(Theme.Color.hairline, lineWidth: 1))
                    } else {
                        CardContainer {
                            Label("Trials you open will appear here.", systemImage: "clock")
                                .font(.subheadline).foregroundStyle(Theme.Color.muted)
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Saved trials").font(.title3.bold())
                        Spacer()
                        if !saved.trials.isEmpty {
                            Button("See all", action: openSaved)
                                .font(.subheadline).foregroundStyle(Theme.Color.accent).frame(minHeight: 44)
                        }
                    }
                    if !saved.trials.isEmpty {
                        VStack(spacing: 0) {
                            ForEach(Array(saved.trials.reversed().prefix(2).enumerated()), id: \.element.id) { index, record in
                                if index > 0 { Divider() }
                                HStack(spacing: 12) {
                                    Image(systemName: "bookmark.fill")
                                        .foregroundStyle(Theme.Color.accent)
                                        .accessibilityHidden(true)
                                    RecentActivityRow(record: record, showsSavedMark: false, onOpen: openSaved)
                                }
                            }
                        }
                        .padding(.horizontal, 16)
                        .background(Theme.Color.paper, in: RoundedRectangle(cornerRadius: Theme.Radius.card))
                        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card).stroke(Theme.Color.hairline, lineWidth: 1))
                    } else {
                        CardContainer {
                            Label("Save trials to compare them or prepare an appointment brief.", systemImage: "bookmark")
                                .font(.subheadline).foregroundStyle(Theme.Color.muted)
                        }
                    }
                }
                // Future: a "Trials near you" MapKit preview can be inserted here.
            }.padding(Theme.Metrics.screenPadding)
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 12) {
                Text(session.user.map { "Welcome, \($0.firstName)" } ?? "Welcome")
                    .font(.largeTitle.bold())
                Text("Here’s where you left off.")
                    .foregroundStyle(.white.opacity(0.85))
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, Theme.Metrics.screenPadding)
            .padding(.top, 16)
            .padding(.bottom, 28)
            .background(Color(red: 37/255, green: 99/255, blue: 235/255))
        }
        .foregroundStyle(Theme.Color.ink)
        .tint(Theme.Color.accent)
        .background(Theme.Color.paper)
        .navigationDestination(isPresented: Binding(
            get: { openedRecent != nil },
            set: { if !$0 { openedRecent = nil } }
        )) {
            if let record = openedRecent, let result = record.recommendation, let profile = record.profile {
                TrialPassportView(profile: profile, result: result, responseSource: record.source ?? "Unknown")
            }
        }
        .navigationTitle("").navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Color(red: 37/255, green: 99/255, blue: 235/255), for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                HStack(spacing: Theme.Spacing.s) {
                    BrandMark(size: 28, color: .white)
                    Text("TrialUnity")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                }
                .accessibilityElement(children: .combine)
            }
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    if let user = session.user {
                        Text(user.name)
                        Text(user.email)
                    }
                    Button("Sign out", systemImage: "rectangle.portrait.and.arrow.right", role: .destructive) { confirmsSignOut = true }
                } label: {
                    Image(systemName: "person.crop.circle")
                        .foregroundStyle(.white)
                        .accessibilityLabel("Account")
                }
            }
        }
        .confirmationDialog("Sign out of TrialUnity?", isPresented: $confirmsSignOut, titleVisibility: .visible) {
            Button("Sign out", role: .destructive) { session.signOut() }
        } message: {
            Text("Your saved trials stay on this device and return when you sign in with the same email.")
        }
    }

    private func journeyValue(
        _ value: String, label: String, icon: String, detail: String, colors: [Color]
    ) -> some View {
        VStack(spacing: 0) {
            Image(systemName: icon)
                .font(.system(size: 42, weight: .regular))
                .foregroundStyle(LinearGradient(colors: colors, startPoint: .topLeading, endPoint: .bottomTrailing))
                .frame(height: 54)
                .padding(.bottom, 12)
                .accessibilityHidden(true)
            Text("\(value) \(label)")
                .font(.subheadline.bold())
                .monospacedDigit()
                .foregroundStyle(Theme.Color.ink)
                .lineLimit(1)
                .minimumScaleFactor(0.65)
            Text(detail)
                .font(.caption)
                .foregroundStyle(Theme.Color.muted)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 6)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 8)
        .padding(.top, 12)
        .padding(.bottom, 16)
        .background(Theme.Color.paper, in: RoundedRectangle(cornerRadius: 14))
        .shadow(color: Theme.Color.accent.opacity(0.10), radius: 12, x: 0, y: 4)
        .accessibilityElement(children: .combine)
    }

}

private struct RecentActivityRow: View {
    @Environment(SavedTrialsStore.self) private var saved
    let record: SavedTrial
    var showsSavedMark = true
    let onOpen: () -> Void

    var body: some View {
        if record.recommendation == nil || record.profile == nil {
            // Older saved snapshot without details: still show the title and ID.
            Button(action: onOpen) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(record.displayTitle).font(.subheadline.weight(.semibold)).lineLimit(2)
                        Text(record.id).font(.provenance).foregroundStyle(Theme.Color.muted)
                    }
                    Spacer(minLength: 0)
                    CardNavigationArrow()
                }
                .foregroundStyle(Theme.Color.ink)
                .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                .padding(.vertical, 14).contentShape(Rectangle())
            }.buttonStyle(.plain)
        } else if let result = record.recommendation, let profile = record.profile {
            Button(action: onOpen) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: Theme.Spacing.xs) {
                            TrialStatusText(status: result.trial.status)
                            Text("·").font(.caption).foregroundStyle(Theme.Color.muted)
                            Text(record.id).font(.provenance).foregroundStyle(Theme.Color.muted)
                            if showsSavedMark, saved.contains(record.id) {
                                Image(systemName: "bookmark.fill").font(.caption2).foregroundStyle(Theme.Color.accent)
                                    .accessibilityLabel("Saved")
                            }
                            if EligibilitySummary(status: result.structured_eligibility?.status) == .unknown {
                                Text("Needs review").font(.caption2.weight(.semibold)).foregroundStyle(Theme.Color.attention)
                            }
                        }
                        Text(record.displayTitle).font(.subheadline.weight(.semibold)).lineLimit(2)
                        Label(PatientPresentation.location(result.trial, near: profile.location) ?? record.id, systemImage: "mappin.and.ellipse")
                            .font(.caption).foregroundStyle(Theme.Color.muted).lineLimit(1)
                    }
                    Spacer(minLength: 0)
                    CardNavigationArrow()
                }
                .foregroundStyle(Theme.Color.ink)
                .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                .padding(.vertical, 14).contentShape(Rectangle())
            }.buttonStyle(.plain)
            .accessibilityHint(showsSavedMark ? "Opens Trial Passport" : "Opens saved trials")
        }
    }
}

private struct RecentActivityView: View {
    @Environment(\.recentTrialActivity) private var activity
    @State private var openedRecent: SavedTrial?

    var body: some View {
        List {
            Section {
                ForEach(activity?.trials ?? []) { record in
                    CardContainer(padding: Theme.Metrics.compactPadding) {
                        RecentActivityRow(record: record) { openedRecent = record }
                    }
                    .listRowInsets(EdgeInsets(top: Theme.Spacing.xs, leading: Theme.Metrics.screenPadding,
                                             bottom: Theme.Spacing.xs, trailing: Theme.Metrics.screenPadding))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                }
            } footer: {
                Text("Your 20 most recently viewed trials.")
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(Theme.Color.paper)
        .navigationDestination(isPresented: Binding(
            get: { openedRecent != nil },
            set: { if !$0 { openedRecent = nil } }
        )) {
            if let record = openedRecent, let result = record.recommendation, let profile = record.profile {
                TrialPassportView(profile: profile, result: result, responseSource: record.source ?? "Unknown")
            }
        }
        .navigationTitle("Recent activity")
        .navigationBarTitleDisplayMode(.inline)
    }
}
