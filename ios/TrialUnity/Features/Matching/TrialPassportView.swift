import SwiftUI

struct TrialPassportView: View {
    @Environment(\.recentTrialActivity) private var recentActivity
    let profile: PatientProfile
    let result: TrialRecommendation
    let responseSource: String
    private var trial: Trial { result.trial }
    private var eligibility: EligibilitySummary { EligibilitySummary(status: result.structured_eligibility?.status) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                BrandedSurface {
                    VStack(alignment: .leading, spacing: 10) {
                        FlowLayout(spacing: Theme.Spacing.xs) {
                            Text(trial.nct_id).font(.caption)
                            Text("·").font(.caption)
                            TrialStatusText(status: trial.status)
                            if let phase = trial.phases.first {
                                Text("·").font(.caption)
                                Text(phase.replacingOccurrences(of: "_", with: " ").capitalized).font(.caption)
                            }
                        }
                        Text(trial.title).font(.title2.bold())
                        if let sponsor = trial.sponsor { Text(sponsor).font(.caption).foregroundStyle(Theme.Color.muted) }
                        if let location = PatientPresentation.location(trial, near: profile.location) {
                            Label(location, systemImage: "mappin.and.ellipse").font(.subheadline)
                        }
                        Text("\(trial.mapSites.count) study locations")
                            .font(.caption).foregroundStyle(Theme.Color.muted)
                        NavigationLink {
                            TrialLocationsView(trial: trial, profileLocation: profile.location)
                        } label: {
                            Label("View trial locations", systemImage: "map")
                                .frame(minHeight: 44)
                        }

                    }
                }
                PassportSection(title: "Your match") {
                    VStack(alignment: .leading, spacing: 12) {
                        ViewThatFits(in: .horizontal) {
                            HStack { matchLabels }
                            VStack(alignment: .leading) { matchLabels }
                        }
                        MatchTraceView(signals: MatchTraceBuilder.build(profile: profile, result: result))
                    }
                }
                PassportSection(title: "At a glance") {
                    GlanceGrid(items: [
                        ("Treatment", trial.interventions.first ?? "Not reported"),
                        ("Phase", trial.phases.first ?? "Not reported"),
                        ("Listed site", PatientPresentation.location(trial, near: profile.location) ?? "Not reported"),
                        ("Status", trial.status.replacingOccurrences(of: "_", with: " ").capitalized)
                    ])
                }
                PassportSection(title: "What to confirm") {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(PatientPresentation.confirmations(result), id: \.self) { item in
                            Label(item, systemImage: "exclamationmark.circle").font(.subheadline)
                        }
                        DisclosureGroup("Review eligibility evidence") {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Age: \(trial.minimum_age ?? "Not reported") – \(trial.maximum_age ?? "Not reported")")
                                Text("Sex requirement: \(trial.sex ?? "Not reported")")
                                ForEach(result.explanation.eligibility_notes + result.explanation.manual_review_signals, id: \.self) { Text($0) }
                                if let criteria = result.structured_eligibility?.criteria {
                                    ForEach(criteria.keys.sorted(), id: \.self) { key in
                                        if let criterion = criteria[key] { Text(criterion.reason) }
                                    }
                                }
                            }.font(.footnote).padding(.top, 8)
                        }.font(.subheadline)
                    }
                }
                AppointmentModeEntry(trials: [SavedTrial(id: trial.nct_id, title: trial.title, savedAt: nil,
                    source: responseSource, profile: profile, recommendation: result)], context: "")
                ESRScoreView(esr: result.esr)
                PassportSection(title: "About this study") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text(PatientPresentation.summary(trial.brief_summary)).font(.subheadline)
                        DisclosureGroup("Read full study description") {
                            Text(trial.brief_summary ?? "Not reported").font(.footnote).padding(.top, 8)
                            Text("Interventions: \(trial.interventions.isEmpty ? "Not reported" : trial.interventions.joined(separator: ", "))").font(.footnote)
                            Text("Sites: \(trial.locations.isEmpty ? "Not reported" : trial.locations.joined(separator: "; "))").font(.footnote)
                        }
                        DisclosureGroup("View full eligibility criteria") { Text(trial.eligibility_criteria ?? "Not reported").font(.footnote).padding(.top, 8) }
                    }
                }
                if let risk = result.representation_risk {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Experimental research insight").font(.caption.bold()).foregroundStyle(Theme.Color.experimental)
                        Text("Predicted representation risk: \(RiskLevelDisplay.label(risk.risk_level))").font(.subheadline)
                        Text("Not observed evidence").font(.caption).foregroundStyle(Theme.Color.muted)
                        DisclosureGroup("Learn more") { RepresentationRiskView(risk: risk).padding(.top, 8) }
                    }
                }
                DisclosureGroup("Source & provenance") {
                    VStack(alignment: .leading, spacing: 8) {
                        if let url = URL(string: trial.source_url ?? "https://clinicaltrials.gov/study/\(trial.nct_id)") { Link("View on ClinicalTrials.gov", destination: url) }
                        Text("Response source: \(responseSource)")
                        if let source = trial.enrollment_sex_source { Text(source) }
                        if let source = trial.enrollment_race_source { Text(source) }
                    }.font(.footnote).padding(.top, 8)
                }
            }.padding(Theme.Metrics.screenPadding)
        }
        .background(Theme.Color.paper)
        .onAppear { recentActivity?.record(result, profile: profile, source: responseSource) }
        .toolbar { SaveTrialButton(result: result, profile: profile, source: responseSource) }
        .navigationTitle("Trial Passport").navigationBarTitleDisplayMode(.inline)
    }
    @ViewBuilder private var matchLabels: some View {
        Text(RelevanceTier(score: result.score).label).font(.headline)
        Label(eligibility.shortLabel, systemImage: eligibility.symbolName).font(.subheadline).foregroundStyle(eligibility.tintColor)
    }
}

#Preview {
    NavigationStack {
        TrialPassportView(
            profile: PatientProfile(draft: .sample),
            result: TrialRecommendation(
                trial: Trial(
                    nct_id: "NCT00000001", title: "Phase 2 Study of Trastuzumab Deruxtecan in Metastatic Breast Cancer",
                    status: "RECRUITING", conditions: ["Metastatic Breast Cancer"], interventions: ["Trastuzumab Deruxtecan"],
                    phases: ["PHASE2"], brief_summary: "A study evaluating trastuzumab deruxtecan in HER2-positive metastatic breast cancer.",
                    eligibility_criteria: "- Must be 18 years or older\n- ECOG status 0-1", sex: "FEMALE", minimum_age: "18 Years", maximum_age: nil,
                    locations: ["Boston, Massachusetts, United States"], sponsor: "Example Sponsor", source_url: "https://clinicaltrials.gov/study/NCT00000001",
                    enrollment_sex_distribution: nil, enrollment_race_distribution: nil,
                    enrollment_sex_source: nil, enrollment_race_source: nil, target_enrollment: 200
                ),
                score: 0.72, score_breakdown: [:],
                relevance: RelevanceScores(overall: 0.72, lexical: 0.6, semantic: 0.7, structured: ["condition": 1, "intervention": 1, "location": 1, "phase": 0], weighted_components: [:], weights: [:]),
                explanation: MatchExplanation(matched_terms: ["breast", "cancer"], eligibility_notes: ["Sex listed by the study: FEMALE."], ranking_rationale: "", patient_friendly_summary: "A study of trastuzumab deruxtecan.", relevant_signals: [], manual_review_signals: ["Full eligibility criteria and site availability require study-team review."]),
                structured_eligibility: StructuredEligibility(status: "compatible", criteria: [
                    "minimum_age": EligibilityCriterion(state: "compatible", reason: "Patient age 58 years; trial minimum age 18 years."),
                    "sex": EligibilityCriterion(state: "compatible", reason: "Patient sex: FEMALE; trial sex: FEMALE."),
                ]),
                esr: ESRResult(score: 63, mode: "mixed", evidence_coverage: 0.6, components: [
                    "socioeconomic": ComponentEvidence(score: 36, evidence_coverage: 1.0, evidence_type: "observed_geographic", rationale: "Site reach and geographic spread only.", source: "ClinicalTrials.gov site locations", missing_evidence: []),
                    "sex": ComponentEvidence(score: 100, evidence_coverage: 1.0, evidence_type: "protocol_inclusivity", rationale: "Protocol eligibility is open to all sexes.", source: "eligibilityModule.sex", missing_evidence: []),
                    "race": ComponentEvidence(score: nil, evidence_coverage: 0.0, evidence_type: "insufficient_data", rationale: "No reported participant race/ethnicity enrollment.", source: nil, missing_evidence: ["Participant race/ethnicity enrollment not reported."]),
                ], weights_used: ["socioeconomic": 0.35, "sex": 0.25, "race": 0.4]),
                representation_risk: nil
            ),
            responseSource: "clinicaltrials.gov"
        ).environment(SavedTrialsStore())
            .environment(AppointmentStore(file: FileManager.default.temporaryDirectory.appendingPathComponent("passport-preview-appointment.json")))
    }
}
