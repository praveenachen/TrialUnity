import SwiftUI
import CryptoKit

struct WelcomeView: View {
    let onStart: () -> Void
    private let blue = Color(red: 37/255, green: 99/255, blue: 235/255)
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            VStack(spacing: 28) {
                AnimatedCareLogo()
                Text("TrialUnity").font(.title.bold())
            }
            .frame(maxWidth: .infinity)
            Spacer()
            Button("Continue", action: onStart)
                .font(.headline).frame(maxWidth: .infinity, minHeight: Theme.Metrics.buttonHeight)
                .foregroundStyle(blue)
                .background(.white, in: RoundedRectangle(cornerRadius: Theme.Radius.control))
                .padding(.bottom, 24)
        }
        .padding(Theme.Metrics.screenPadding).foregroundStyle(.white)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(blue.ignoresSafeArea())
    }
}

/// A hand cradling a heart, inside a white disc. On launch a ring draws itself
/// once around the mark, then the crescent, hand and heart settle in. No looping;
/// shown fully drawn under Reduce Motion.
private struct AnimatedCareLogo: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var ring: CGFloat = 0
    @State private var disc = false
    @State private var crescent = false
    @State private var hand = false
    @State private var heart = false

    private let dark = Color(red: 37/255, green: 68/255, blue: 154/255)
    private let light = Color(red: 74/255, green: 154/255, blue: 219/255)
    private let soft = Color(red: 108/255, green: 155/255, blue: 211/255)

    var body: some View {
        ZStack {
            Circle().stroke(.white.opacity(0.18), lineWidth: 3).frame(width: 196, height: 196)
            Circle().trim(from: 0, to: ring)
                .stroke(.white, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .frame(width: 196, height: 196)
                .rotationEffect(.degrees(-90))

            Circle().fill(.white).frame(width: 164, height: 164)
                .scaleEffect(disc ? 1 : 0.85).opacity(disc ? 1 : 0)

            ZStack {
                CrescentShape().fill(light)
                    .rotationEffect(.degrees(crescent ? 0 : -40), anchor: .center)
                    .opacity(crescent ? 1 : 0)
                Group {
                    HandShape().fill(dark)
                    FingerShape().fill(dark)
                }
                .offset(y: hand ? 0 : 14).opacity(hand ? 1 : 0)
                ZStack {
                    HeartShape().fill(dark).frame(width: 40, height: 38).offset(y: 2)
                    HeartShape().fill(soft).frame(width: 27, height: 25).offset(y: 4)
                }
                .scaleEffect(heart ? 1 : 0.2).opacity(heart ? 1 : 0)
            }
            .frame(width: 100, height: 100)
            .scaleEffect(1.15)
        }
        .frame(width: 200, height: 200)
        .onAppear(perform: play)
        .accessibilityHidden(true)
    }

    private func play() {
        guard !reduceMotion else { ring = 1; disc = true; crescent = true; hand = true; heart = true; return }
        withAnimation(.easeInOut(duration: 1.3)) { ring = 1 }
        withAnimation(.easeOut(duration: 0.5).delay(0.6)) { disc = true }
        withAnimation(.easeOut(duration: 0.7).delay(0.9)) { crescent = true }
        withAnimation(.spring(response: 0.6, dampingFraction: 0.8).delay(1.1)) { hand = true }
        withAnimation(.spring(response: 0.5, dampingFraction: 0.55).delay(1.5)) { heart = true }
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
            p.addCurve(to: CGPoint(x: 70, y: 98), control1: CGPoint(x: 8, y: 90), control2: CGPoint(x: 45, y: 106))
            p.addCurve(to: CGPoint(x: 92, y: 40), control1: CGPoint(x: 90, y: 88), control2: CGPoint(x: 98, y: 62))
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
                    HStack(alignment: .center, spacing: 4) {
                        journeyTile(value: count.map(String.init) ?? "—", label: "Matches", symbol: "magnifyingglass", tint: Theme.Color.accent)
                        journeyConnector
                        journeyTile(value: String(saved.trials.count), label: "Saved", symbol: "bookmark.fill", tint: Theme.Color.evidence)
                        journeyConnector
                        let briefs = activity?.briefIDs.count ?? 0
                        Button(action: openSaved) {
                            journeyTile(value: briefs > 0 ? String(briefs) : "Prepare",
                                        label: briefs > 0 ? "Briefs" : "Start brief →",
                                        symbol: "doc.text", tint: Theme.Color.violet, compactValue: briefs == 0)
                        }.buttonStyle(.plain)
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

                if !saved.trials.isEmpty {
                    ShortlistCard(trials: saved.trials, openSaved: openSaved)
                } else {
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

    private func journeyTile(value: String, label: String, symbol: String, tint: Color, compactValue: Bool = false) -> some View {
        VStack(spacing: 6) {
            Image(systemName: symbol).font(.caption.weight(.bold)).foregroundStyle(tint)
            Text(value).font(compactValue ? .subheadline.bold() : .title2.bold()).monospacedDigit()
                .foregroundStyle(tint).lineLimit(1).minimumScaleFactor(0.8)
                .frame(minHeight: 28)
            Text(label).font(.caption).foregroundStyle(Theme.Color.muted).lineLimit(1).minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(tint.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }

    private var journeyConnector: some View {
        Image(systemName: "chevron.right").font(.caption2.weight(.bold))
            .foregroundStyle(Theme.Color.muted.opacity(0.5)).accessibilityHidden(true)
    }
}

/// A compact snapshot of the saved shortlist, from real saved-trial snapshots only.
/// Metrics that can't be known reliably are omitted rather than estimated.
private struct ShortlistCard: View {
    let trials: [SavedTrial]
    let openSaved: () -> Void

    private var strong: Int { trials.filter { $0.recommendation.map { RelevanceTier(score: $0.score) == .strong } ?? false }.count }
    private var needsReview: Int {
        trials.filter { $0.recommendation.map { EligibilitySummary(status: $0.structured_eligibility?.status) != .compatible } ?? false }.count
    }
    /// Only when the search profile had a location and a listed site text-matches it.
    private var nearby: Int? {
        let withLocation = trials.filter { $0.recommendation != nil && !($0.profile?.location ?? "").isEmpty }
        guard !withLocation.isEmpty else { return nil }
        return withLocation.filter { record in
            guard let loc = record.profile?.location, let trial = record.recommendation?.trial else { return false }
            return trial.locations.contains { $0.localizedCaseInsensitiveContains(loc) }
        }.count
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("YOUR SHORTLIST").font(.caption.weight(.semibold)).foregroundStyle(Theme.Color.muted)
            Text(trials.count == 1 ? "1 saved trial" : "\(trials.count) saved trials").font(.title3.bold())
            HStack(alignment: .top, spacing: 8) {
                metric(strong, "Strong\nmatches", "checkmark.circle.fill", Theme.Color.accent)
                metric(needsReview, "Review\nneeded", "questionmark.circle.fill", Theme.Color.attention)
                if let nearby { metric(nearby, "Nearby\nsite", "mappin.circle.fill", Theme.Color.evidence) }
            }
            Button(action: openSaved) {
                HStack {
                    Text("View saved trials")
                    Spacer()
                    Image(systemName: "arrow.right")
                }
                .font(.subheadline.weight(.semibold)).foregroundStyle(Theme.Color.accent).frame(minHeight: 44)
            }
        }
        .padding(Theme.Metrics.cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.card))
        .accessibilityElement(children: .contain)
    }

    private func metric(_ value: Int, _ label: String, _ symbol: String, _ tint: Color) -> some View {
        VStack(spacing: 4) {
            Image(systemName: symbol).foregroundStyle(tint).accessibilityHidden(true)
            Text("\(value)").font(.title2.bold()).monospacedDigit().foregroundStyle(tint)
            Text(label).font(.caption).foregroundStyle(Theme.Color.muted).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(tint.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
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
