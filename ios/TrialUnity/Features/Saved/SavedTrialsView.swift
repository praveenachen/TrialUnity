import SwiftUI

struct SaveTrialButton: View {
    @Environment(SavedTrialsStore.self) private var store
    let result: TrialRecommendation
    let profile: PatientProfile
    let source: String

    private var isSaved: Bool { store.contains(result.id) }

    var body: some View {
        Button(isSaved ? "Unsave" : "Save", systemImage: isSaved ? "bookmark.fill" : "bookmark") {
            if isSaved { store.remove(result.id) } else { store.save(result, profile: profile, source: source) }
            Haptics.saveToggled()
        }
        .tint(Theme.Color.accent)
    }
}

struct SavedTrialsView: View {
    @Environment(SavedTrialsStore.self) private var store
    @State private var selection = TrialSelection()
    @State private var selectionMessage: String?
    @State private var openedTrial: SavedTrial?
    private var selected: [SavedTrial] { store.trials.filter { selection.ids.contains($0.id) } }

    var body: some View {
        List {
            if !store.trials.isEmpty {
                Section {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        Text("\(store.trials.count) trials").font(.title2.bold()).foregroundStyle(Theme.Color.ink)
                        Text("Select 1–3 trials for an appointment, or 2–3 to compare.")
                        if let selectionMessage {
                            Label(selectionMessage, systemImage: "info.circle")
                                .foregroundStyle(Theme.Color.attention)
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(Theme.Color.muted)
                }
                .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets(top: Theme.Spacing.s, leading: Theme.Metrics.screenPadding,
                                         bottom: Theme.Spacing.s, trailing: Theme.Metrics.screenPadding))

                ForEach(store.trials) { record in
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        SelectionCheckRow(isSelected: selection.ids.contains(record.id)) {
                            let accepted = selection.toggle(record.id)
                            selectionMessage = accepted ? nil : "Select no more than 3 trials. Deselect one first."
                            Haptics.selectionChanged()
                        }
                        .frame(minHeight: Theme.Metrics.minTapTarget, alignment: .leading)

                        if record.recommendation != nil, record.profile != nil {
                            Button {
                                openedTrial = record
                            } label: {
                                SavedTrialRow(record: record, showsDisclosure: true)
                            }
                            .buttonStyle(.pressable)
                            .accessibilityHint("Opens Trial Passport")
                        } else {
                            SavedTrialRow(record: record)
                        }
                    }
                    .listRowInsets(EdgeInsets(top: Theme.Spacing.xs, leading: Theme.Metrics.screenPadding, bottom: Theme.Spacing.xs, trailing: Theme.Metrics.screenPadding))
                    .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets(top: Theme.Spacing.s, leading: Theme.Metrics.screenPadding,
                                         bottom: Theme.Spacing.s, trailing: Theme.Metrics.screenPadding))
                    .listRowSeparator(.hidden)
                    .swipeActions(edge: .trailing) {
                        Button("Remove", systemImage: "trash", role: .destructive) {
                            store.remove(record.id)
                        }
                    }
                    .contextMenu {
                        Button("Remove from saved", systemImage: "trash", role: .destructive) {
                            store.remove(record.id)
                        }
                    }
                }

            }
        }
        .overlay {
            if store.trials.isEmpty {
                ContentUnavailableView("No saved trials", systemImage: "bookmark",
                                       description: Text("Save trials from your results."))
            }
        }
        .navigationTitle("Saved trials")
        .navigationDestination(isPresented: Binding(
            get: { openedTrial != nil },
            set: { if !$0 { openedTrial = nil } }
        )) {
            if let record = openedTrial, let result = record.recommendation, let profile = record.profile {
                TrialPassportView(profile: profile, result: result, responseSource: record.source ?? "Unknown")
            }
        }
        .scrollContentBackground(.hidden)
        .listStyle(.plain)
        .listRowInsets(EdgeInsets(top: Theme.Spacing.s, leading: Theme.Metrics.screenPadding,
                                 bottom: Theme.Spacing.s, trailing: Theme.Metrics.screenPadding))
        .background(Theme.Color.paper)
        .onChange(of: store.trials.map(\.id)) { _, ids in selection.retain(Set(ids)) }
        .safeAreaInset(edge: .bottom) {
            if !selection.ids.isEmpty {
                selectionActionBar
            }
        }
    }

    private var selectionActionBar: some View {
        VStack(spacing: Theme.Spacing.s) {
            Text("\(selection.ids.count) selected")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.Color.muted)
            HStack(spacing: Theme.Spacing.s) {
                NavigationLink {
                    TrialComparisonView(trials: selected)
                } label: {
                    Text("Compare")
                }
                .buttonStyle(AppointmentEntryButtonStyle())
                .disabled(!selection.canCompare)

                AppointmentModeEntry(trials: selected, context: "")
            }
        }
        .padding(.horizontal, Theme.Metrics.screenPadding)
        .padding(.top, Theme.Spacing.s)
        .padding(.bottom, Theme.Spacing.xs)
        .background(.bar)
    }
}

/// A stacked, phone-friendly comparison of 2-3 saved trials: each trial gets a
/// clear identity (a lettered badge + title + NCT ID) once at the top, then every
/// field is one aligned section with a row per trial, tagged by the same badge --
/// no spreadsheet-style horizontal scrolling.
struct TrialComparisonView: View {
    let trials: [SavedTrial]

    private let badges = ["A", "B", "C"]

    var body: some View {
        if !(2...3).contains(trials.count) {
            ContentUnavailableView(
                "Select 2\u{2013}3 saved trials",
                systemImage: "square.on.square",
                description: Text("Go back to Saved trials and select 2 or 3 trials to compare.")
            )
            .background(Theme.Color.paper)
        } else {
            VerticalScrollView {
                VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                    BrandedSurface {
                        VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                            ForEach(Array(trials.enumerated()), id: \.element.id) { index, trial in
                                HStack(alignment: .top, spacing: Theme.Spacing.s) {
                                    badge(index)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(trial.displayTitle)
                                            .font(.subheadline.weight(.semibold))
                                            .foregroundStyle(Theme.Color.ink)
                                        ProvenanceText(text: trial.id)
                                    }
                                }
                            }
                        }
                    }

                    ForEach(["Clinical relevance", "Structured eligibility", "Interventions", "Phase", "Locations", "ESR", "Experimental predicted representation risk"], id: \.self) { field in
                        VStack(alignment: .leading, spacing: 10) {
                            Text(field == "ESR" ? "Representation & access" : field).font(.headline)
                            ForEach(Array(trials.enumerated()), id: \.element.id) { index, trial in
                                HStack(alignment: .top) {
                                    badge(index)
                                    if field == "Experimental predicted representation risk" {
                                        Text(trial.recommendation?.representation_risk.map { "Predicted: \($0.risk_level ?? "Unknown") · not observed evidence" } ?? "Not reported")
                                    } else {
                                        Text(SavedTrialPresentation.fields(trial).first { $0.id == field }?.value ?? "Not reported").lineLimit(3)
                                    }
                                }.font(.subheadline)
                            }
                            DisclosureGroup("Details") {
                                ForEach(Array(trials.enumerated()), id: \.element.id) { index, trial in
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text(badges[index]).font(.headline)
                                        if field == "ESR" { ESRScoreView(esr: trial.recommendation?.esr) }
                                        else if field == "Structured eligibility" {
                                            ForEach(SavedTrialPresentation.fields(trial).filter { ["Age range", "Sex requirement"].contains($0.id) }) { Text("\($0.id): \($0.value)") }
                                        } else { Text(SavedTrialPresentation.fields(trial).first { $0.id == field }?.value ?? "Not reported") }
                                    }.font(.footnote).padding(.top, 8)
                                }
                            }.font(.subheadline)
                        }
                        Divider()
                    }
                }
                .padding(Theme.Metrics.screenPadding)
            }
            .background(Theme.Color.paper)
            .navigationTitle("Compare trials")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func badge(_ index: Int) -> some View {
        Text(badges[index])
            .font(.caption.weight(.bold))
            .foregroundStyle(.white)
            .frame(width: 22, height: 22)
            .background(Theme.Color.accent, in: Circle())
    }
}

struct AppointmentBriefView: View {
    @Environment(\.recentTrialActivity) private var activity
    @State private var activityID = UUID()
    let trials: [SavedTrial]
    let context: String
    private var shareText: String { AppointmentBrief.generate(trials, context: context) }

    var body: some View {
        VerticalScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                    Text("Appointment brief")
                        .font(.editorialTitle)
                        .foregroundStyle(Theme.Color.ink)
                    Text("\(trials.count) trial\(trials.count == 1 ? "" : "s") to discuss")
                        .font(.subheadline)
                        .foregroundStyle(Theme.Color.muted)
                    if !context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Text("Patient context: \(context)")
                            .font(.footnote)
                            .foregroundStyle(Theme.Color.muted)
                    }
                }

                ForEach(Array(trials.prefix(3).enumerated()), id: \.element.id) { index, trial in
                    AppointmentBriefTrialSection(index: index + 1, trial: trial)
                }

                ShareLink(item: shareText) {
                    Text("Share appointment brief")
                        .font(.headline)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity, minHeight: Theme.Metrics.buttonHeight)
                        .background(Theme.Color.accent, in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
                }
                .buttonStyle(.pressable)

                Text(AppointmentBrief.disclaimer)
                    .font(.caption)
                    .foregroundStyle(Theme.Color.muted)
            }
            .padding(Theme.Metrics.screenPadding)
        }
        .background(Theme.Color.paper)
        .onAppear { activity?.recordBrief(activityID) }
        .navigationTitle("Appointment brief")
        .navigationBarTitleDisplayMode(.inline)

    }
}

/// One trial's structured brief block: why it surfaced, what to confirm with
/// the care team, and questions worth asking -- never raw internal metrics
/// like lexical/semantic relevance scores.
private struct AppointmentBriefTrialSection: View {
    let index: Int
    let trial: SavedTrial

    private var eligibility: [(label: String, criterion: EligibilityCriterion)] {
        guard let criteria = trial.recommendation?.structured_eligibility?.criteria else { return [] }
        let labels: [String: String] = [
            "minimum_age": "Minimum age criterion", "maximum_age": "Maximum age criterion",
            "sex": "Sex criterion", "recruitment_status": "Recruitment status",
        ]
        return criteria.keys.sorted().compactMap { key in
            guard let c = criteria[key] else { return nil }
            return (labels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized, c)
        }
    }

    private var questions: [String] { AppointmentBrief.questions(for: trial) }

    var body: some View {
        CardContainer {
            VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                Text("TRIAL \(String(format: "%02d", index))")
                    .font(.sectionLabel)
                    .foregroundStyle(Theme.Color.muted)
                    .tracking(0.5)

                VStack(alignment: .leading, spacing: 2) {
                    Text(trial.displayTitle)
                        .font(.editorialHeadline)
                        .foregroundStyle(Theme.Color.ink)
                    ProvenanceText(text: trial.id)
                }

                briefBlock(title: "Why it surfaced") {
                    Text(trial.recommendation.map { RelevanceTier(score: $0.score).label } ?? "Relevance unknown")
                }
                briefBlock(title: "Things to confirm") {
                    bulletList(trial.recommendation.map { PatientPresentation.confirmations($0) } ?? ["Full eligibility and site availability"])
                }

                briefBlock(title: "Questions to ask") {
                    bulletList(questions)
                }

                if let result = trial.recommendation, let profile = trial.profile {
                    NavigationLink("View trial") { TrialPassportView(profile: profile, result: result, responseSource: trial.source ?? "Unknown") }
                }
                if let url = URL(string: trial.sourceLink) {
                    Link(destination: url) {
                        Label("View source", systemImage: "arrow.up.right.square")
                            .font(.caption.weight(.medium))
                    }
                    .tint(Theme.Color.accent)
                }
            }
        }
    }

    private func briefBlock<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(title.uppercased())
                .font(.sectionLabel)
                .foregroundStyle(Theme.Color.muted)
            content()
        }
    }

    private func bulletList(_ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            ForEach(items, id: \.self) { item in
                Label(item, systemImage: "square")
                    .font(.subheadline)
                    .foregroundStyle(Theme.Color.ink)
            }
        }
    }
}
