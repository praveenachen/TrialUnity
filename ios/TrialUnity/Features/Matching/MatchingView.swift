import SwiftUI

struct MatchingView: View {
    @State private var model: MatchingModel
    @State private var attempt = 0

    init(profile: PatientProfile) {
        _model = State(initialValue: MatchingModel(profile: profile))
    }

    var body: some View {
        List {
            switch model.state {
            case .idle, .loading:
                ProgressView("Finding trials for you…")
            case .loaded(let response):
                ForEach(response.results) { result in
                    NavigationLink {
                        RecommendationDetailView(result: result)
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(result.trial.title).font(.headline)
                            Text("\(result.trial.nct_id) · \(result.trial.status)")
                            Text("Relevance: \(result.score, format: .number.precision(.fractionLength(3)))")
                            if let location = result.trial.locations.first { Text(location) }
                        }
                        .font(.subheadline)
                    }
                }
            case .empty:
                Text("No trials were found for this profile. You can edit your profile or retry.")
                Button("Retry", action: retry)
            case .failed(let error):
                Text(error.localizedDescription)
                Button("Retry", action: retry)
            }
            #if DEBUG
            Section("Development provenance") {
                Text("API: \(APIConfiguration.current.baseURL?.absoluteString ?? "Not configured")")
                if let source = model.source { Text("Source: \(source)") }
            }.font(.caption).foregroundStyle(Theme.Color.muted)
            #endif
        }
        .scrollContentBackground(.hidden)
        .background(Theme.Color.paper)
        .navigationTitle("Matching trials")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: attempt) { await model.load() }
    }

    private func retry() {
        model.prepareRetry()
        attempt += 1
    }
}

private struct RecommendationDetailView: View {
    let result: TrialRecommendation

    var body: some View {
        List {
            Section(result.trial.nct_id) { Text(result.trial.title) }
            Section("Relevance") {
                Text("Relative ranking score: \(result.score, format: .number.precision(.fractionLength(3)))")
                scores(result.score_breakdown)
                Text(result.explanation.ranking_rationale)
            }
            Section("Structured eligibility") {
                Text(result.structured_eligibility?.status ?? "Unknown")
                if let eligibility = result.structured_eligibility {
                    ForEach(eligibility.criteria.keys.sorted(), id: \.self) { key in
                        if let criterion = eligibility.criteria[key] {
                            Text("\(key): \(criterion.state) — \(criterion.reason)")
                        }
                    }
                }
                Text("Structured checks are not full medical eligibility.")
            }
            Section("ESR · representation and access evidence") {
                if let esr = result.esr {
                    Text("Score: \(number(esr.score)) / 100")
                    Text("Evidence coverage: \(esr.evidence_coverage, format: .percent)")
                    ForEach(esr.components.keys.sorted(), id: \.self) { key in
                        if let component = esr.components[key] {
                            VStack(alignment: .leading) {
                                Text("\(key): \(number(component.score))")
                                Text("Coverage: \(component.evidence_coverage, format: .percent)")
                                Text(component.rationale)
                                #if DEBUG
                                Text("Evidence: \(component.evidence_type)")
                                Text("Source: \(component.source ?? "Unavailable")")
                                #endif
                            }
                        }
                    }
                    #if DEBUG
                    Text("ESR evidence mode: \(esr.mode)")
                    #endif
                } else { Text("Evidence unavailable") }
            }
            if let risk = result.representation_risk {
                Section("Experimental predicted representation risk") {
                    Text("Predicted risk: \(risk.risk_level ?? "Unavailable")")
                    Text("Confidence: \(number(risk.confidence))")
                    Text("Prediction only; not observed ESR evidence.")
                    ForEach(risk.limitations, id: \.self) { Text($0) }
                    #if DEBUG
                    Text("Evidence type: \(risk.evidence_type)")
                    Text("Model: \(risk.model_version)")
                    #endif
                }
            }
        }
        .scrollContentBackground(.hidden)
        .background(Theme.Color.paper)
        .navigationTitle("Trial details")
    }

    private func number(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3))) } ?? "Unavailable"
    }

    private func scores(_ values: [String: Double]) -> some View {
        ForEach(values.keys.sorted(), id: \.self) { key in
            Text("\(key): \(number(values[key]))")
        }
    }
}
