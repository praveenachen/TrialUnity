import SwiftUI
import CryptoKit

struct WelcomeView: View {
    let onStart: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var logo = false
    @State private var wordmark = false
    @State private var button = false
    @State private var spotlight = false

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
                .buttonStyle(.pressable)
                .padding(.bottom, compact ? 30 : 40)
                .opacity(button ? 1 : 0)
            }
            .padding(.horizontal, 32)
            .background {
                RadialGradient(
                    colors: [Color(red: 0, green: 112/255, blue: 1), splashBlue],
                    center: UnitPoint(x: spotlight ? 0.65 : 0.35, y: spotlight ? 0.32 : 0.52),
                    startRadius: 0,
                    endRadius: CGFloat(VisualNumber.dimension(Double(geometry.size.height))) * 0.48
                )
                .ignoresSafeArea()
            }
        }
        .environment(\.colorScheme, .dark)
        .onAppear(perform: play)
        .onChange(of: reduceMotion) { _, reduced in
            if reduced {
                var transaction = Transaction(); transaction.disablesAnimations = true
                withTransaction(transaction) { spotlight = false; logo = true; wordmark = true; button = true }
            } else { play() }
        }
    }

    private func play() {
        guard !reduceMotion else { logo = true; wordmark = true; button = true; return }
        withAnimation(.easeInOut(duration: 4).repeatForever(autoreverses: true)) { spotlight = true }
        withAnimation(.easeOut(duration: 0.8)) { logo = true }
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

    /// Name only: the email is the key for this device's per-user data, so it stays fixed.
    func updateName(_ name: String) {
        guard let current = user else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let updated = LocalUser(name: trimmed, email: current.email)
        defaults.set(try? JSONEncoder().encode(updated), forKey: Self.key)
        user = updated
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
        GeometryReader { geometry in
            ScrollView {
                VStack(spacing: Theme.Metrics.sectionSpacing) {
                    VStack(spacing: Theme.Spacing.s) {
                        BrandMark(size: 64, color: .white)
                        Text("Sign in").font(.largeTitle.bold()).padding(.top, Theme.Spacing.s)
                    }
                    .frame(maxWidth: .infinity)

                    VStack(alignment: .leading, spacing: Theme.Spacing.l) {
                        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                            Text("Name").font(.subheadline.weight(.semibold))
                            OutlinedTextField(placeholder: "Your name", text: $name)
                                .foregroundStyle(Theme.Color.ink)
                                .textContentType(.name).submitLabel(.next)
                                .focused($focus, equals: .name)
                                .onSubmit { focus = .email }
                        }
                        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                            Text("Email").font(.subheadline.weight(.semibold))
                            OutlinedTextField(placeholder: "you@example.com", text: $email, keyboardType: .emailAddress)
                                .foregroundStyle(Theme.Color.ink)
                                .textContentType(.emailAddress).textInputAutocapitalization(.never).autocorrectionDisabled()
                                .submitLabel(.go)
                                .focused($focus, equals: .email)
                                .onSubmit(submit)
                        }
                    }
                    Button(action: submit) {
                        Text("Continue")
                            .font(.headline)
                            .foregroundStyle(Theme.Color.accent)
                            .frame(maxWidth: .infinity, minHeight: Theme.Metrics.buttonHeight)
                            .background(.white.opacity(isValid ? 1 : 0.65),
                                        in: RoundedRectangle(cornerRadius: Theme.Radius.control))
                    }
                    .buttonStyle(.pressable)
                    .disabled(!isValid)
                }
                .frame(maxWidth: 440)
                .padding(Theme.Metrics.screenPadding)
                .frame(maxWidth: .infinity)
                .frame(minHeight: CGFloat(VisualNumber.dimension(Double(geometry.size.height))), alignment: .center)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .foregroundStyle(.white)
        .tint(Theme.Color.accent)
        .background(Theme.Color.accent.ignoresSafeArea())
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
    @Environment(AppointmentStore.self) private var appointments
    @Environment(\.openAppointments) private var openAppointments
    @Environment(UserSession.self) private var session
    @Environment(SavedTrialsStore.self) private var saved
    @Environment(\.recentTrialActivity) private var activity
    @State private var openedRecent: SavedTrial?
    let draft: PatientProfileDraft
    let count: Int?
    let explore: () -> Void
    let openSaved: () -> Void
    let openLatestSearch: () -> Void
    let openMenu: () -> Void

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
                }.buttonStyle(.pressable)

                VStack(alignment: .leading, spacing: 14) {
                    Text("Your trial journey")
                        .font(.title3.bold())
                        .foregroundStyle(Theme.Color.ink)
                    HStack(alignment: .top, spacing: 10) {
                        Button(action: openLatestSearch) {
                            journeyValue(count.flatMap { $0 > 0 ? String($0) : nil } ?? "", label: "Matches",
                                         icon: "magnifyingglass", detail: "View matching\ntrials",
                                         colors: [.pink, .purple, .blue])
                        }.accessibilityHint("Opens your most recent search")
                        Button(action: openSaved) {
                            journeyValue(saved.trials.isEmpty ? "" : String(saved.trials.count), label: "Saved",
                                         icon: "bookmark", detail: "Bookmarked\nfor later",
                                         colors: [.cyan, .blue, .purple])
                        }.accessibilityHint("Opens saved trials")
                        Button(action: openAppointments) {
                            journeyValue(appointments.records.isEmpty ? "" : String(appointments.records.count), label: "Briefs",
                                         icon: "doc.text", detail: "Summary ready\nto review",
                                         colors: [.pink, .purple, .cyan])
                        }.accessibilityHint("Opens all appointment records")
                    }.buttonStyle(.pressable)
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
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, Theme.Metrics.screenPadding)
            .padding(.top, 16)
            .padding(.bottom, 28)
            .background(Color(red: 37/255, green: 99/255, blue: 235/255).ignoresSafeArea(edges: .top))
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
        // Transparent bar over the blue header block, so there is no seam between them.
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button(action: openMenu) {
                    Image(systemName: "person.crop.circle")
                        .font(.title3)
                        .foregroundStyle(.white)
                        .frame(minWidth: 44, minHeight: 44)
                }
                .accessibilityLabel("Account menu")
            }
            ToolbarItem(placement: .topBarTrailing) {
                HStack(spacing: Theme.Spacing.s) {
                    Text("TrialUnity")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                    BrandMark(size: 28, color: .white)
                }
                .accessibilityElement(children: .combine)
            }
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
            Text(value.isEmpty ? label : "\(value) \(label)")
                .font(.subheadline.bold())
                .monospacedDigit()
                .foregroundStyle(Theme.Color.ink)
                .lineLimit(1)
                .minimumScaleFactor(0.65)

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

/// Slide-in account sidebar, opened from the Home account button.
struct SideMenuView: View {
    let user: LocalUser
    let openAppointments: () -> Void
    let openSaved: () -> Void
    let openHistory: () -> Void
    let openSettings: () -> Void
    let signOut: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                Image(systemName: "person.crop.circle.fill")
                    .font(.system(size: 48)).foregroundStyle(Theme.Color.accent)
                Text(user.name).font(.title3.bold()).foregroundStyle(Theme.Color.ink)
                Text(user.email).font(.subheadline).foregroundStyle(Theme.Color.muted).lineLimit(1)
            }
            .padding(.top, 72).padding(.bottom, Theme.Spacing.xl)
            Divider()
            row("Appointments", "calendar", openAppointments)
            row("Saved trials", "bookmark", openSaved)
            row("Search history", "clock.arrow.circlepath", openHistory)
            row("Settings", "gearshape", openSettings)
            Spacer()
            Divider()
            row("Sign out", "rectangle.portrait.and.arrow.right", signOut, tint: Theme.Color.conflict)
                .padding(.bottom, Theme.Spacing.l)
        }
        .padding(.horizontal, Theme.Metrics.screenPadding)
        .frame(width: 290, alignment: .leading)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(Theme.Color.paper.ignoresSafeArea())
        .shadow(color: .black.opacity(0.18), radius: 16, x: 4)
        .accessibilityElement(children: .contain)
    }

    private func row(_ title: String, _ symbol: String, _ action: @escaping () -> Void, tint: Color = Theme.Color.ink) -> some View {
        Button(action: action) {
            HStack(spacing: Theme.Spacing.m) {
                Image(systemName: symbol).frame(width: 24).foregroundStyle(tint == Theme.Color.ink ? Theme.Color.accent : tint)
                Text(title).font(.body)
                Spacer()
            }
            .foregroundStyle(tint)
            .frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.pressable)
    }
}

/// Local-only settings: the display name shown on Home. The email is fixed because
/// it keys this device's saved data.
struct SettingsView: View {
    @Environment(UserSession.self) private var session
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Name", text: $name).textContentType(.name)
                    LabeledContent("Email", value: session.user?.email ?? "")
                }
            }
            .navigationTitle("Settings").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { session.updateName(name); dismiss() }
                        .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .onAppear { name = session.user?.name ?? "" }
        }
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
            }.buttonStyle(.pressable)
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
            }.buttonStyle(.pressable)
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
