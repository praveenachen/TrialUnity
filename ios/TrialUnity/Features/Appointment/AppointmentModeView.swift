import SwiftUI

/// Shared entry point: resuming never overwrites an existing session implicitly.
struct AppointmentModeEntry: View {
    @Environment(AppointmentStore.self) private var store
    let trials: [SavedTrial]
    let context: String
    var showsResume = true
    @State private var opensMode = false
    @State private var replacesSession = false

    var body: some View {
        VStack(spacing: 8) {
            if showsResume, store.session != nil {
                Button(store.session?.completed == true ? "View completed appointment" : "Continue Appointment Mode") { opensMode = true }
                    .frame(minHeight: 44)
            }
            if !trials.isEmpty {
                Button("Start Appointment Mode") {
                    if store.session != nil { replacesSession = true }
                    else { opensMode = store.start(trials, context: context) }
                }
                .frame(minHeight: 44)
                .disabled(!(1...3).contains(Set(trials.map(\.id)).count))
            }
            if let message = store.message { Text(message).font(.caption).foregroundStyle(Theme.Color.attention) }
        }
        .navigationDestination(isPresented: $opensMode) { AppointmentModeView() }
        .confirmationDialog("Start a new appointment?", isPresented: $replacesSession, titleVisibility: .visible) {
            Button("Replace previous appointment", role: .destructive) { opensMode = store.start(trials, context: context) }
            Button("Cancel", role: .cancel) {}
        } message: { Text("This replaces the locally saved appointment and its notes. Share its summary first if you want to keep a copy.") }
    }
}

struct AppointmentModeView: View {
    @Environment(AppointmentStore.self) private var store
    @FocusState private var editingNotes: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                if let session = store.session {
                    if session.completed { completion(session) }
                    else if session.trials.indices.contains(session.currentIndex) {
                        let trial = session.trials[session.currentIndex]
                        Text("Trial \(session.currentIndex + 1) of \(session.trials.count)").font(.headline)
                        ProvenanceText(text: trial.id)
                        Text(trial.displayTitle).font(.title2.bold())
                        Text("Checked means Discussed only—not eligible, confirmed, or resolved.")
                            .font(.caption).foregroundStyle(Theme.Color.muted)
                        discussionBlock("Things to confirm", trial: trial, questions: false, session: session)
                        discussionBlock("Questions to ask", trial: trial, questions: true, session: session)
                        Text("Notes").font(.headline)
                        TextEditor(text: Binding(
                            get: { store.session?.notes[trial.id] ?? "" },
                            set: { store.setNote($0, trialID: trial.id) }
                        ))
                        .focused($editingNotes)
                        .frame(minHeight: 160)
                        .padding(8)
                        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control).stroke(Theme.Color.hairline))
                        .accessibilityLabel("Notes for \(trial.displayTitle)")
                        HStack(spacing: 16) {
                            Button("Previous") { editingNotes = false; store.move(to: session.currentIndex - 1) }
                                .disabled(session.currentIndex == 0).frame(minHeight: 44)
                            Spacer()
                            Button(session.currentIndex == session.trials.count - 1 ? "Complete appointment" : "Next") {
                                editingNotes = false
                                store.reviewCurrent()
                                if session.currentIndex == session.trials.count - 1 { store.complete() }
                                else { store.move(to: session.currentIndex + 1) }
                            }.frame(minHeight: 44)
                        }
                    }
                    Text(AppointmentBrief.disclaimer).font(.caption).foregroundStyle(Theme.Color.muted)
                } else { Text("Select 1–3 saved trials to start an appointment.") }
                if let message = store.message { Text(message).foregroundStyle(Theme.Color.attention).font(.footnote) }
            }.padding(Theme.Metrics.screenPadding)
        }
        .background(Theme.Color.paper)
        .navigationTitle("Appointment Mode").navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { editingNotes = false }
            }
        }
    }
    private func discussionBlock(_ title: String, trial: SavedTrial, questions: Bool, session: AppointmentSession) -> some View {
        CardContainer {
            VStack(alignment: .leading, spacing: 8) {
                Text(title.uppercased()).font(.sectionLabel).foregroundStyle(Theme.Color.muted)
                ForEach(AppointmentBrief.discussionItems(for: trial).filter { $0.isQuestion == questions }) { item in
                    let checked = session.isDiscussed(item.id, trialID: trial.id)
                    Button {
                        store.toggle(item, trialID: trial.id)
                        if store.message == nil { Haptics.selectionChanged() }
                    } label: {
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: checked ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(checked ? Theme.Color.accent : Theme.Color.muted)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.text).foregroundStyle(Theme.Color.ink)
                                if checked { Text("Discussed").font(.caption).foregroundStyle(Theme.Color.muted) }
                            }
                            Spacer(minLength: 0)
                        }.frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(item.text)
                    .accessibilityValue(checked ? "Discussed" : "Not discussed")
                    .accessibilityHint("Double tap to toggle discussion status only")
                }
            }
        }
    }
    private func completion(_ session: AppointmentSession) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Appointment complete").font(.title2.bold())
            Text("\(session.reviewedIDs.intersection(Set(session.selectedIDs)).count) trials reviewed")
            Text("\(session.discussedQuestionCount) questions discussed")
            Text("\(session.noteCount) notes added")
            NavigationLink("View summary") {
                ScrollView {
                    Text(AppointmentBrief.summary(session))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(Theme.Metrics.screenPadding)
                }
                .background(Theme.Color.paper)
                .navigationTitle("Appointment summary")
                .toolbar { ShareLink(item: AppointmentBrief.summary(session)) { Label("Share summary", systemImage: "square.and.arrow.up") } }
            }.frame(minHeight: 44)
            ShareLink(item: AppointmentBrief.summary(session)) { Label("Share summary", systemImage: "square.and.arrow.up") }.frame(minHeight: 44)
            Button("Review appointment") { store.reopen() }.frame(minHeight: 44)
        }
    }
}
