import SwiftUI

private struct OpenAppointmentDetailsKey: EnvironmentKey {
    static let defaultValue: (UUID) -> Void = { _ in }
}

private struct OpenAppointmentsKey: EnvironmentKey {
    static let defaultValue: () -> Void = {}
}

extension EnvironmentValues {
    var openAppointments: () -> Void {
        get { self[OpenAppointmentsKey.self] }
        set { self[OpenAppointmentsKey.self] = newValue }
    }
    var openAppointmentDetails: (UUID) -> Void {
        get { self[OpenAppointmentDetailsKey.self] }
        set { self[OpenAppointmentDetailsKey.self] = newValue }
    }
}

struct AppointmentEntryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .multilineTextAlignment(.center)
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, minHeight: Theme.Metrics.buttonHeight)
            .padding(.horizontal, 8)
            .background(Theme.Color.accent, in: RoundedRectangle(cornerRadius: Theme.Radius.control))
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.45)
    }
}

private struct AppointmentActionStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, minHeight: 60)
            .padding(.horizontal, 8)
            .background(Theme.Color.accent.opacity(configuration.isPressed ? 0.75 : 1),
                        in: RoundedRectangle(cornerRadius: Theme.Radius.control))
    }
}

private struct AppointmentActionLabel: View {
    let title: String
    let symbol: String
    var body: some View {
        VStack(spacing: 5) {
            Image(systemName: symbol)
            Text(title).fixedSize(horizontal: false, vertical: true)
        }.padding(.vertical, 8)
    }
}

/// Starting a new appointment preserves every previous record in the Appointments tab.
struct AppointmentModeEntry: View {
    @Environment(AppointmentStore.self) private var store
    let trials: [SavedTrial]
    let context: String
    @State private var opensMode = false

    var body: some View {
        VStack(spacing: 8) {
            if !trials.isEmpty {
                Button("Appointment Mode") { opensMode = store.start(trials, context: context) }
                    .buttonStyle(AppointmentEntryButtonStyle())
                    .disabled(!(1...3).contains(Set(trials.map(\.id)).count))
            }
            if let message = store.message { Text(message).font(.caption).foregroundStyle(Theme.Color.attention) }
        }
        .navigationDestination(isPresented: $opensMode) { AppointmentModeView() }
    }
}

struct AppointmentsView: View {
    @Environment(AppointmentStore.self) private var store

    var body: some View {
        List {
            if store.records.isEmpty {
                ContentUnavailableView("No appointments yet", systemImage: "calendar", description: Text("Select trials in Saved or open a trial passport to start Appointment Mode."))
            }
            ForEach(store.records.reversed()) { record in
                NavigationLink(value: record.id) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(record.session.context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Appointment" : record.session.context)
                            .font(.headline)
                        Text(record.session.completed ? "Completed" : "In progress")
                            .font(.subheadline).foregroundStyle(Theme.Color.accent)
                        if let date = record.createdAt {
                            Text("Started \(date.formatted(date: .abbreviated, time: .shortened))")
                        } else { Text("Start date not recorded") }
                        if let date = record.completedAt {
                            Text("Completed \(date.formatted(date: .abbreviated, time: .shortened))")
                        } else if let date = record.updatedAt {
                            Text("Updated \(date.formatted(date: .abbreviated, time: .shortened))")
                        }
                        Text(record.session.trials.map(\.displayTitle).joined(separator: " · "))
                            .lineLimit(2)
                    }.font(.caption).padding(.vertical, 4)
                }
            }
            if let message = store.message { Text(message).foregroundStyle(Theme.Color.attention) }
        }
        .scrollContentBackground(.hidden)
        .background(Theme.Color.paper)
        .navigationTitle("Appointments")
        .navigationDestination(for: UUID.self) { AppointmentDetailsView(recordID: $0) }
    }
}

struct AppointmentModeView: View {
    @Environment(AppointmentStore.self) private var store
    @Environment(\.openAppointmentDetails) private var openDetails
    @Environment(\.openAppointments) private var openAppointments
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    var recordID: UUID? = nil
    @State private var openedRecordID: UUID?
    private var selectedRecordID: UUID? { recordID ?? openedRecordID }
    @FocusState private var editingNotes: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                if let session = store.record(selectedRecordID)?.session {
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
                            get: { store.record(selectedRecordID)?.session.notes[trial.id] ?? "" },
                            set: { store.setNote($0, trialID: trial.id, recordID: selectedRecordID) }
                        ))
                        .focused($editingNotes)
                        .frame(minHeight: 160)
                        .padding(8)
                        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control).stroke(Theme.Color.hairline))
                        .accessibilityLabel("Notes for \(trial.displayTitle)")
                        HStack(spacing: 16) {
                            Button("Previous") { editingNotes = false; store.move(to: session.currentIndex - 1, recordID: selectedRecordID) }
                                .disabled(session.currentIndex == 0).frame(minHeight: 44)
                            Spacer()
                            Button(session.currentIndex == session.trials.count - 1 ? "Complete appointment" : "Next Trial Info") {
                                editingNotes = false
                                store.reviewCurrent(recordID: selectedRecordID)
                                if session.currentIndex == session.trials.count - 1 { store.complete(recordID: selectedRecordID) }
                                else { store.move(to: session.currentIndex + 1, recordID: selectedRecordID) }
                            }.frame(minHeight: 44)
                        }
                    }
                    Text(AppointmentBrief.disclaimer).font(.caption).foregroundStyle(Theme.Color.muted)
                } else { Text("Select 1–3 saved trials to start an appointment.") }
                if let message = store.message { Text(message).foregroundStyle(Theme.Color.attention).font(.footnote) }
            }.padding(Theme.Metrics.screenPadding)
        }
        .background(Theme.Color.paper)
        .onAppear { if openedRecordID == nil { openedRecordID = store.record(recordID)?.id } }
        .navigationTitle(store.record(selectedRecordID)?.session.completed == true ? "Appointment Summary" : "Appointment Mode").navigationBarTitleDisplayMode(.inline)
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
                        store.toggle(item, trialID: trial.id, recordID: selectedRecordID)
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
        VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Appointment Summary").font(.title2.bold())
                Text("Saved in Appointments. You can return to your notes and discussion details anytime.")
                    .font(.subheadline).foregroundStyle(Theme.Color.muted)
            }
            CardContainer {
                VStack(alignment: .leading, spacing: 12) {
                    Label("\(session.reviewedIDs.intersection(Set(session.selectedIDs)).count) trials reviewed", systemImage: "doc.text")
                    Label("\(session.discussedQuestionCount) questions discussed", systemImage: "checkmark.bubble")
                    Label("\(session.noteCount) notes added", systemImage: "note.text")
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            let layout = dynamicTypeSize.isAccessibilitySize ? AnyLayout(VStackLayout(spacing: 10)) : AnyLayout(HStackLayout(spacing: 10))
            layout {
                Button {
                    if let id = store.record(selectedRecordID)?.id {
                        dismiss()
                        openDetails(id)
                    }
                } label: { AppointmentActionLabel(title: "View details", symbol: "arrow.right") }
                Button {
                    store.reopen(recordID: selectedRecordID)
                    store.move(to: 0, recordID: selectedRecordID)
                } label: { AppointmentActionLabel(title: "Edit details", symbol: "arrow.left") }
                ShareLink(item: AppointmentBrief.summary(session)) {
                    AppointmentActionLabel(title: "Share", symbol: "square.and.arrow.up")
                }
            }.buttonStyle(AppointmentActionStyle())
            Button {
                dismiss()
                openAppointments()
            } label: {
                Label("Finished", systemImage: "checkmark")
            }.buttonStyle(AppointmentEntryButtonStyle())
        }
    }
}

/// The Appointments tab is the single home for saved discussion details.
struct AppointmentDetailsView: View {
    @Environment(AppointmentStore.self) private var store
    let recordID: UUID
    @State private var editing = false

    var body: some View {
        ScrollView {
            if let record = store.record(recordID) {
                VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                    Text(record.session.completed ? "Completed appointment" : "Appointment in progress")
                        .font(.headline)
                    if let date = record.createdAt {
                        Text(date.formatted(date: .abbreviated, time: .shortened))
                            .font(.subheadline).foregroundStyle(Theme.Color.muted)
                    }
                    HStack(spacing: 10) {
                        Button {
                            if record.session.completed {
                                store.reopen(recordID: recordID)
                                store.move(to: 0, recordID: recordID)
                            }
                            editing = store.message == nil
                        } label: {
                            AppointmentActionLabel(title: record.session.completed ? "Edit details" : "Continue", symbol: "arrow.left")
                        }
                        ShareLink(item: AppointmentBrief.summary(record.session)) {
                            AppointmentActionLabel(title: "Share", symbol: "square.and.arrow.up")
                        }
                    }.buttonStyle(AppointmentActionStyle())
                    Text("Discussion summary").font(.title3.bold())
                    Text("Discussed does not mean eligible, confirmed, or resolved.")
                        .font(.caption).foregroundStyle(Theme.Color.muted)
                    ForEach(record.session.trials) { trial in
                        AppointmentTrialSummary(trial: trial, session: record.session)
                    }
                    DisclosureGroup("Original trial evidence and context") {
                        Text(AppointmentBrief.generate(record.session.trials, context: record.session.context))
                            .font(.footnote)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                            .padding(.top, 8)
                    }.font(.subheadline.weight(.semibold))
                    Text(AppointmentBrief.disclaimer)
                        .font(.caption).foregroundStyle(Theme.Color.muted)
                    if let message = store.message { Text(message).foregroundStyle(Theme.Color.attention) }
                }.padding(Theme.Metrics.screenPadding)
            }
        }
        .background(Theme.Color.paper)
        .navigationTitle("Appointment Details")
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(isPresented: $editing) { AppointmentModeView(recordID: recordID) }
    }
}

private struct AppointmentTrialSummary: View {
    let trial: SavedTrial
    let session: AppointmentSession
    private var items: [AppointmentItem] { AppointmentBrief.discussionItems(for: trial) }
    private var discussed: [AppointmentItem] { items.filter { session.isDiscussed($0.id, trialID: trial.id) } }
    private var open: [AppointmentItem] { items.filter { !session.isDiscussed($0.id, trialID: trial.id) } }

    var body: some View {
        CardContainer {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(trial.displayTitle).font(.headline)
                    Text(trial.id).font(.caption).foregroundStyle(Theme.Color.muted)
                }
                Divider()
                DisclosureGroup("Discussed items (\(discussed.count))") { bulletList(discussed) }
                DisclosureGroup("Open items (\(open.count))") { bulletList(open) }
                Divider()
                Text("Your notes").font(.subheadline.weight(.semibold))
                let notes = session.notes[trial.id]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                Text(notes.isEmpty ? "No notes added." : notes)
                    .font(.subheadline)
                    .textSelection(.enabled)
                if let url = URL(string: trial.sourceLink) {
                    Link(destination: url) { Label("View trial source", systemImage: "arrow.up.right.square") }
                        .font(.subheadline).frame(minHeight: 44)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func bulletList(_ items: [AppointmentItem]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if items.isEmpty {
                Text("None").foregroundStyle(Theme.Color.muted)
            }
            ForEach(items) { item in
                HStack(alignment: .top, spacing: 8) {
                    Text("•").accessibilityHidden(true)
                    Text(item.text).frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }.font(.subheadline).padding(.top, 8).textSelection(.enabled)
    }
}
