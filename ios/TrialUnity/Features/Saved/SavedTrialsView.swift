import SwiftUI

struct SaveTrialButton: View {
    @Environment(SavedTrialsStore.self) private var store
    let result: TrialRecommendation
    let profile: PatientProfile
    let source: String
    var body: some View {
        Button(store.contains(result.id) ? "Unsave" : "Save", systemImage: store.contains(result.id) ? "bookmark.fill" : "bookmark") {
            if store.contains(result.id) { store.remove(result.id) }
            else { store.save(result, profile: profile, source: source) }
        }
    }
}

struct SavedTrialsView: View {
    @Environment(SavedTrialsStore.self) private var store
    @State private var selection = TrialSelection()
    @State private var selectionMessage: String?
    @State private var context = ""
    private var selected: [SavedTrial] { store.trials.filter { selection.ids.contains($0.id) } }

    var body: some View {
        List {
            if store.trials.isEmpty {
                ContentUnavailableView("No saved trials", systemImage: "bookmark", description: Text("Save trials from your results or Trial Passport to revisit them here."))
            } else {
                Section {
                    Text("Saved snapshots may be out of date. Relevance reflects the original search profile; confirm current details with the study team.")
                    Text("Select up to 3 trials. Comparison needs at least 2.")
                    if let selectionMessage { Text(selectionMessage).foregroundStyle(.secondary) }
                }
                ForEach(store.trials) { record in
                    Section {
                        if let date = record.savedAt { Text("Saved \(date.formatted(date: .abbreviated, time: .omitted))").font(.caption) }
                        if let result = record.recommendation, let profile = record.profile {
                            NavigationLink {
                                TrialPassportView(profile: profile, result: result, responseSource: record.source ?? "Unknown")
                            } label: {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(record.displayTitle).font(.headline)
                                    Text(record.id)
                                    Text(result.trial.status)
                                    Text(result.trial.locations.first ?? "Location not reported")
                                    Text(RelevanceTier(score: result.score).label)
                                    Text("ESR: \(SavedTrialPresentation.number(result.esr?.score))")
                                }
                            }
                        } else {
                            Text(record.displayTitle)
                            Text(record.id)
                            Text("Details unavailable. Find this trial in a new search and save it again.")
                        }
                        Button(selection.ids.contains(record.id) ? "Deselect" : "Select") {
                            selectionMessage = selection.toggle(record.id) ? nil : "Select no more than 3 trials. Deselect one first."
                        }
                        Button("Remove from saved", role: .destructive) { store.remove(record.id) }
                    }
                }
                Section("Next steps") {
                    NavigationLink("Compare selected (\(selected.count))") { TrialComparisonView(trials: selected) }
                        .disabled(!selection.canCompare)
                    TextField("Condition or context to share (optional)", text: $context, axis: .vertical)
                    Text("Only include details you want to share. Original search conditions and backend eligibility explanations are also included.").font(.caption)
                    NavigationLink("Create appointment brief") {
                        AppointmentBriefView(trials: selected, context: context)
                    }.disabled(selected.isEmpty)
                }
            }
        }
        .navigationTitle("Saved trials")
        .scrollContentBackground(.hidden)
        .background(Theme.Color.paper)
        .onChange(of: store.trials.map(\.id)) { _, ids in selection.retain(Set(ids)) }
    }
}

struct TrialComparisonView: View {
    let trials: [SavedTrial]
    var body: some View {
        List {
            if !(2...3).contains(trials.count) {
                Text("Select 2–3 saved trials to compare.")
            } else {
                Section { Text("Snapshots from original searches. Relevance may reflect different profiles. ESR evidence and experimental predictions are separate.") }
                ForEach(SavedTrialPresentation.fields(trials[0])) { field in
                    Section(field.id) {
                        ForEach(trials) { trial in
                            VStack(alignment: .leading, spacing: 6) {
                                Text("\(trial.displayTitle) · \(trial.id)").font(.headline)
                                Text(SavedTrialPresentation.fields(trial).first { $0.id == field.id }?.value ?? "Not reported")
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Compare trials")
        .scrollContentBackground(.hidden)
        .background(Theme.Color.paper)
    }
}

struct AppointmentBriefView: View {
    let trials: [SavedTrial]
    let context: String
    private var text: String { AppointmentBrief.generate(trials, context: context) }
    var body: some View {
        ScrollView { Text(text).textSelection(.enabled).padding() }
            .navigationTitle("Appointment brief")
            .background(Theme.Color.paper)
            .toolbar { ShareLink(item: text) { Label("Share", systemImage: "square.and.arrow.up") } }
    }
}
