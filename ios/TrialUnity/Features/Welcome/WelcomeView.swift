import SwiftUI
import CryptoKit

struct WelcomeView: View {
    let onStart: () -> Void
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            VStack(spacing: 24) {
                AnimatedCareLogo()
                Text("TrialUnity").font(.title.bold()).foregroundStyle(Color(red: 37/255, green: 68/255, blue: 154/255))
            }
            .frame(maxWidth: .infinity)
            Spacer()
            PrimaryButton(title: "Continue", action: onStart)
                .padding(.bottom, 24)
        }
        .padding(Theme.Metrics.screenPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.white.ignoresSafeArea())
    }
}

/// The hand-and-heart mark on white, like the original logo. On launch the
/// crescent sweeps around once, the hand rises and the heart pops in. No looping;
/// shown fully drawn under Reduce Motion.
private struct AnimatedCareLogo: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var sweep: CGFloat = 0
    @State private var hand = false
    @State private var heart = false

    private let dark = Color(red: 37/255, green: 68/255, blue: 154/255)
    private let light = Color(red: 74/255, green: 154/255, blue: 219/255)
    private let soft = Color(red: 108/255, green: 155/255, blue: 211/255)

    var body: some View {
        ZStack {
            CrescentShape().fill(light)
                .mask {
                    // A thick arc that grows clockwise from the crescent's lower tip.
                    Circle().trim(from: 0, to: sweep * 0.7)
                        .stroke(lineWidth: 100)
                        .frame(width: 60, height: 60)
                        .rotationEffect(.degrees(105))
                }
            Group {
                HandShape().fill(dark)
                FingerShape().fill(dark)
            }
            .offset(y: hand ? 0 : 12).opacity(hand ? 1 : 0)
            ZStack {
                HeartShape().fill(dark).frame(width: 40, height: 38).offset(y: 2)
                HeartShape().fill(soft).frame(width: 27, height: 25).offset(y: 4)
            }
            .scaleEffect(heart ? 1 : 0.2).opacity(heart ? 1 : 0)
        }
        .frame(width: 100, height: 100)
        .scaleEffect(2.3)
        .frame(width: 230, height: 230)
        .onAppear(perform: play)
        .accessibilityHidden(true)
    }

    private func play() {
        guard !reduceMotion else { sweep = 1; hand = true; heart = true; return }
        withAnimation(.easeInOut(duration: 1.2).delay(0.2)) { sweep = 1 }
        withAnimation(.spring(response: 0.7, dampingFraction: 0.8).delay(0.6)) { hand = true }
        withAnimation(.spring(response: 0.5, dampingFraction: 0.55).delay(1.3)) { heart = true }
    }
}

/// Paths are authored in a 100x100 space and scaled to the frame.
private func scaled(_ rect: CGRect, _ build: (inout Path) -> Void) -> Path {
    var p = Path(); build(&p)
    return p.applying(CGAffineTransform(scaleX: rect.width / 100, y: rect.height / 100))
}

/// Light-blue crescent sweeping around the left and top.
struct CrescentShape: Shape {
    func path(in rect: CGRect) -> Path {
        scaled(rect) { p in
            p.move(to: CGPoint(x: 28, y: 78))
            p.addCurve(to: CGPoint(x: 67, y: 24), control1: CGPoint(x: 0, y: 58), control2: CGPoint(x: 24, y: 4))
            p.addCurve(to: CGPoint(x: 28, y: 78), control1: CGPoint(x: 42, y: 22), control2: CGPoint(x: 26, y: 54))
            p.closeSubpath()
        }
    }
}

/// Dark-blue open hand cradling from the bottom-left up to the fingers at top right.
struct HandShape: Shape {
    func path(in rect: CGRect) -> Path {
        scaled(rect) { p in
            p.move(to: CGPoint(x: 3, y: 58))
            p.addCurve(to: CGPoint(x: 70, y: 98), control1: CGPoint(x: 8, y: 88), control2: CGPoint(x: 45, y: 102))
            p.addCurve(to: CGPoint(x: 92, y: 40), control1: CGPoint(x: 87, y: 95), control2: CGPoint(x: 98, y: 66))
            p.addCurve(to: CGPoint(x: 78, y: 4), control1: CGPoint(x: 88, y: 24), control2: CGPoint(x: 86, y: 8))
            p.addCurve(to: CGPoint(x: 84, y: 40), control1: CGPoint(x: 76, y: 14), control2: CGPoint(x: 86, y: 28))
            p.addCurve(to: CGPoint(x: 72, y: 70), control1: CGPoint(x: 84, y: 54), control2: CGPoint(x: 80, y: 64))
            p.addCurve(to: CGPoint(x: 10, y: 66), control1: CGPoint(x: 52, y: 88), control2: CGPoint(x: 22, y: 84))
            p.addCurve(to: CGPoint(x: 3, y: 58), control1: CGPoint(x: 6, y: 64), control2: CGPoint(x: 4, y: 62))
            p.closeSubpath()
        }
    }
}

/// Second, shorter finger beside the first.
struct FingerShape: Shape {
    func path(in rect: CGRect) -> Path {
        scaled(rect) { p in
            p.move(to: CGPoint(x: 62, y: 14))
            p.addCurve(to: CGPoint(x: 79, y: 56), control1: CGPoint(x: 70, y: 22), control2: CGPoint(x: 77, y: 40))
            p.addCurve(to: CGPoint(x: 72, y: 34), control1: CGPoint(x: 74, y: 50), control2: CGPoint(x: 72, y: 42))
            p.addCurve(to: CGPoint(x: 62, y: 14), control1: CGPoint(x: 71, y: 26), control2: CGPoint(x: 66, y: 20))
            p.closeSubpath()
        }
    }
}

struct HeartShape: Shape {
    func path(in rect: CGRect) -> Path {
        scaled(rect) { p in
            p.move(to: CGPoint(x: 50, y: 92))
            p.addCurve(to: CGPoint(x: 4, y: 34), control1: CGPoint(x: 28, y: 72), control2: CGPoint(x: 4, y: 56))
            p.addCurve(to: CGPoint(x: 50, y: 26), control1: CGPoint(x: 4, y: 10), control2: CGPoint(x: 40, y: 6))
            p.addCurve(to: CGPoint(x: 96, y: 34), control1: CGPoint(x: 60, y: 6), control2: CGPoint(x: 96, y: 10))
            p.addCurve(to: CGPoint(x: 50, y: 92), control1: CGPoint(x: 96, y: 56), control2: CGPoint(x: 72, y: 72))
            p.closeSubpath()
        }
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
                VStack(alignment: .leading, spacing: 8) {
                    BrandHeader()
                    Text(session.user.map { "Welcome, \($0.firstName)" } ?? "Welcome").font(.largeTitle.bold())
                    Text("Here’s where you left off.").foregroundStyle(Theme.Color.muted)
                }
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
                    Text("YOUR TRIAL JOURNEY").font(.caption.weight(.semibold)).foregroundStyle(Theme.Color.muted)
                    HStack(alignment: .top, spacing: 8) {
                        journeyValue(count.map(String.init) ?? "—", label: "Matches", tint: Theme.Color.accent)
                        journeyValue(String(saved.trials.count), label: "Saved", tint: Theme.Color.evidence)
                        journeyValue(String(activity?.briefIDs.count ?? 0), label: "Briefs", tint: .purple)
                    }
                }
                .padding(Theme.Metrics.cardPadding)
                .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.card))

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
                        Text("\(saved.trials.count)").font(.headline).foregroundStyle(Theme.Color.accent)
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        if let recent = saved.trials.last {
                            Button(action: openSaved) {
                                HStack(spacing: 12) {
                                    Image(systemName: "bookmark.fill")
                                        .foregroundStyle(Theme.Color.accent)
                                        .accessibilityHidden(true)
                                    Text(recent.displayTitle)
                                        .font(.subheadline.weight(.semibold)).lineLimit(2)
                                    Spacer(minLength: 0)
                                    CardNavigationArrow()
                                }.frame(minHeight: 44)
                            }.buttonStyle(.plain)
                            if let result = recent.recommendation {
                                if let location = PatientPresentation.location(result.trial, near: recent.profile?.location) {
                                    Text(location).font(.caption).foregroundStyle(Theme.Color.muted).lineLimit(2)
                                }
                            } else {
                                Text(recent.id).font(.caption).foregroundStyle(Theme.Color.muted)
                            }
                        } else {
                            Text("Save trials to compare them or prepare an appointment brief.")
                                .font(.subheadline).foregroundStyle(Theme.Color.muted)
                        }
                        Button(action: openSaved) {
                            HStack {
                                Text("Open saved trials")
                                Spacer()
                                if saved.trials.isEmpty { Image(systemName: "arrow.right") }
                            }.font(.subheadline.weight(.semibold)).foregroundStyle(Theme.Color.accent).frame(minHeight: 44)
                        }
                    }
                    .padding(Theme.Metrics.cardPadding)
                    .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.card))
                    .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card).stroke(Theme.Color.accent.opacity(0.2), lineWidth: 1))
                }
                // Future: a "Trials near you" MapKit preview can be inserted here.
            }.padding(Theme.Metrics.screenPadding)
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
        .navigationTitle("Home").navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    if let user = session.user {
                        Text(user.name)
                        Text(user.email)
                    }
                    Button("Sign out", systemImage: "rectangle.portrait.and.arrow.right", role: .destructive) { confirmsSignOut = true }
                } label: {
                    Image(systemName: "person.crop.circle").accessibilityLabel("Account")
                }
            }
        }
        .confirmationDialog("Sign out of TrialUnity?", isPresented: $confirmsSignOut, titleVisibility: .visible) {
            Button("Sign out", role: .destructive) { session.signOut() }
        } message: {
            Text("Your saved trials stay on this device and return when you sign in with the same email.")
        }
    }

    private func journeyValue(_ value: String, label: String, tint: Color) -> some View {
        VStack(spacing: 4) {
            Text(value).font(.title2.bold()).monospacedDigit().foregroundStyle(tint)
            Text(label).font(.caption).foregroundStyle(Theme.Color.muted)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }
}

private struct RecentActivityRow: View {
    @Environment(SavedTrialsStore.self) private var saved
    let record: SavedTrial
    let onOpen: () -> Void

    var body: some View {
        if let result = record.recommendation, let profile = record.profile {
            Button(action: onOpen) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: Theme.Spacing.xs) {
                            TrialStatusText(status: result.trial.status)
                            Text("·").font(.caption).foregroundStyle(Theme.Color.muted)
                            Text(record.id).font(.provenance).foregroundStyle(Theme.Color.muted)
                            if saved.contains(record.id) {
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
            .accessibilityHint("Opens Trial Passport")
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
