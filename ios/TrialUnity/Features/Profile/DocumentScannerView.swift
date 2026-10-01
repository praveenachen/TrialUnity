import SwiftUI
import VisionKit
import Vision
import AVFoundation

struct DocumentScannerView: View {
    let draft: PatientProfileDraft
    @Environment(\.dismiss) private var dismiss
    @State private var active = true
    @State private var scanning = false
    @State private var processing = false
    @State private var candidates: [ScanCandidate] = []
    @State private var selected: Set<String> = []
    @State private var message: String?
    @State private var hasScanned = false
    @State private var worker: Task<[ScanCandidate], Error>?
    private var chosen: [ScanCandidate] { candidates.filter { selected.contains($0.id) } }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                    Text(hasScanned && !candidates.isEmpty ? "We found a few details" : "Scan a medical document")
                        .font(.title2.bold())
                    if processing {
                        ProgressView("Reading document…").frame(maxWidth: .infinity).padding()
                    } else {
                        if let message { Text(message).foregroundStyle(Theme.Color.attention) }
                        if hasScanned && candidates.isEmpty {
                            Text("No profile details were confidently detected.").font(.headline)
                        }
                        ForEach(candidates) { item in
                            CardContainer {
                                Toggle(isOn: Binding(get: { selected.contains(item.id) }, set: { on in
                                    if on {
                                        if item.field == .condition || item.field == .age {
                                            for other in candidates where other.field == item.field { selected.remove(other.id) }
                                        }
                                        selected.insert(item.id)
                                    } else { selected.remove(item.id) }
                                })) {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text(item.field.rawValue).font(.caption).foregroundStyle(Theme.Color.muted)
                                        Text(item.value).font(.headline)
                                        if item.needsReview { Text("Needs review").font(.caption).foregroundStyle(Theme.Color.attention) }
                                    }
                                }.frame(minHeight: 44)
                                .accessibilityLabel("Add \(item.field.rawValue): \(item.value)")
                            }
                        }
                        if !candidates.isEmpty {
                            Text("Check each detail — scans can be misread.")
                                .font(.footnote).foregroundStyle(Theme.Color.muted)
                            PrimaryButton(title: "Keep selected suggestions", isEnabled: !selected.isEmpty) {
                                draft.keepScanSuggestions(chosen)
                                candidates = []; selected = []
                                dismiss()
                            }
                        }
                        Button(hasScanned ? "Scan again" : "Open scanner", systemImage: "doc.viewfinder", action: startScan)
                            .frame(maxWidth: .infinity, minHeight: 44).buttonStyle(.borderedProminent)
                        Button("Continue manually") { dismiss() }.frame(maxWidth: .infinity, minHeight: 44)
                    }
                }.padding(Theme.Metrics.screenPadding)
            }
            .background(Theme.Color.paper)
            .navigationTitle("Profile assist").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { worker?.cancel(); dismiss() } } }
            .fullScreenCover(isPresented: $scanning) {
                NativeDocumentScanner { scan in
                    scanning = false
                    guard let scan else { dismiss(); return }
                    read(scan)
                } failed: {
                    scanning = false
                    message = "The document could not be scanned. Try again or continue manually."
                }
            }

        }
        .onDisappear { if !scanning { active = false; worker?.cancel(); worker = nil } }
    }
    private func startScan() {
        guard VNDocumentCameraViewController.isSupported else {
            message = "Document scanning is unavailable on this device. Continue entering your profile manually."
            return
        }
        Task { @MainActor in
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            let allowed: Bool
            if status == .notDetermined { allowed = await AVCaptureDevice.requestAccess(for: .video) }
            else { allowed = status == .authorized }
            guard active else { return }
            if allowed { scanning = true; message = nil }
            else { message = "Camera access is unavailable. You can enable it in Settings or continue manually." }
        }
    }
    private func read(_ scan: VNDocumentCameraScan) {
        processing = true; message = nil; candidates = []; selected = []
        let task = Task.detached(priority: .userInitiated) { () throws -> [ScanCandidate] in
            var lines: [ScanTextLine] = []
            for page in 0..<scan.pageCount {
                try Task.checkCancellation()
                try autoreleasepool {
                    guard let image = scan.imageOfPage(at: page).cgImage else { return }
                    let request = VNRecognizeTextRequest()
                    request.recognitionLevel = .accurate
                    request.recognitionLanguages = ["en-US"]
                    request.usesLanguageCorrection = false
                    try VNImageRequestHandler(cgImage: image).perform([request])
                    for observation in request.results ?? [] {
                        if let text = observation.topCandidates(1).first {
                            lines.append(ScanTextLine(text: text.string, confidence: text.confidence, box: observation.boundingBox, page: page))
                        }
                    }
                }
            }
            try Task.checkCancellation()
            return DocumentProfileAssist.extract(lines)
        }
        worker = task
        Task { @MainActor in
            do {
                let result = try await task.value
                if !task.isCancelled { candidates = result; hasScanned = true }
            } catch {
                if !task.isCancelled { message = "Text could not be read. Try a clearer scan or continue manually." }
            }
            processing = false; worker = nil
        }
    }
}

private struct NativeDocumentScanner: UIViewControllerRepresentable {
    let finished: (VNDocumentCameraScan?) -> Void
    let failed: () -> Void
    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }
    func makeUIViewController(context: Context) -> VNDocumentCameraViewController {
        let controller = VNDocumentCameraViewController()
        controller.delegate = context.coordinator
        return controller
    }
    func updateUIViewController(_ controller: VNDocumentCameraViewController, context: Context) {}
    final class Coordinator: NSObject, VNDocumentCameraViewControllerDelegate {
        let parent: NativeDocumentScanner
        init(parent: NativeDocumentScanner) { self.parent = parent }
        func documentCameraViewControllerDidCancel(_ controller: VNDocumentCameraViewController) { parent.finished(nil) }
        func documentCameraViewController(_ controller: VNDocumentCameraViewController, didFinishWith scan: VNDocumentCameraScan) { parent.finished(scan) }
        func documentCameraViewController(_ controller: VNDocumentCameraViewController, didFailWithError error: Error) { parent.failed() }
    }
}

struct ScanSuggestionModule: View {
    @Bindable var draft: PatientProfileDraft
    let field: ScanCandidate.Field
    @State private var replacement: ScanCandidate?
    private var pending: [ScanCandidate] {
        draft.scanSuggestions.filter { $0.field == field && !draft.acceptedScanSuggestionIDs.contains($0.id) }
    }
    var body: some View {
        ForEach(pending) { item in
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                Text("Suggested from document").font(.caption.weight(.semibold)).foregroundStyle(Theme.Color.accent)
                Text(item.value).font(.headline)
                if item.needsReview { Text("Needs review").font(.caption).foregroundStyle(Theme.Color.attention) }
                Button("Use suggestion") {
                    if !draft.useScanSuggestion(item.id) { replacement = item }
                }.buttonStyle(.borderedProminent).frame(minHeight: 44)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(Theme.Spacing.m)
            .background(Theme.Color.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: Theme.Radius.card))
        }
        .confirmationDialog("Replace your existing value?", isPresented: Binding(
            get: { replacement != nil }, set: { if !$0 { replacement = nil } }
        ), titleVisibility: .visible) {
            Button("Use suggestion instead") {
                if let item = replacement { draft.useScanSuggestion(item.id, replaceExisting: true) }
                replacement = nil
            }
            Button("Keep my value", role: .cancel) { replacement = nil }
        } message: {
            if let item = replacement {
                Text("\(field == .condition ? draft.condition : draft.ageText) → \(item.value)")
            }
        }
    }
}
