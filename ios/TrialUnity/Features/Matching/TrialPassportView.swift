import SwiftUI

/// A trial's full-screen passport: identity, your match, overview, eligibility
/// evidence, representation & access, predicted risk, and source -- pushed onto
/// the navigation stack, not a modal. Every value here comes straight from the
/// backend's TrialRecommendation; nothing is recalculated in Swift.
struct TrialPassportView: View {
    let profile: PatientProfile
    let result: TrialRecommendation
    let responseSource: String

    private var trial: Trial { result.trial }
    private var signals: [MatchSignal] { MatchTraceBuilder.build(profile: profile, result: result) }
    private var eligibility: EligibilitySummary { EligibilitySummary(status: result.structured_eligibility?.status) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.l) {
                identitySection
                yourMatchSection
                overviewSection
                eligibilityEvidenceSection
                representationSection
                if let risk = result.representation_risk {
                    PassportSection(title: "Predicted representation risk") {
                        RepresentationRiskView(risk: risk)
                    }
                }
                sourceSection
            }
            .padding(Theme.Spacing.l)
        }
        .background(Theme.Color.paper)
        .toolbar { SaveTrialButton(result: result, profile: profile, source: responseSource) }
        .navigationTitle("Trial Passport")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: A. Trial identity

    private var identitySection: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.s) {
            Text(trial.title)
                .font(.editorialTitle)
                .foregroundStyle(Theme.Color.ink)

            ProvenanceText(text: trial.nct_id, color: Theme.Color.ink)

            HStack(spacing: Theme.Spacing.s) {
                StatusPill(text: trial.status.capitalized, symbolName: "circle.fill", tint: statusTint)
                if let phase = trial.phases.first {
                    StatusPill(text: phase.replacingOccurrences(of: "_", with: " ").capitalized, symbolName: "chart.bar", tint: Theme.Color.muted)
                }
            }

            if let sponsor = trial.sponsor {
                Text("Sponsor: \(sponsor)")
                    .font(.footnote)
                    .foregroundStyle(Theme.Color.muted)
            }
            if let location = trial.locations.first {
                Label(location, systemImage: "mappin.and.ellipse")
                    .font(.footnote)
                    .foregroundStyle(Theme.Color.muted)
            }
        }
    }

    private var statusTint: Color {
        trial.status.uppercased() == "RECRUITING" ? Theme.Color.accent : Theme.Color.muted
    }

    // MARK: B. Your Match

    private var yourMatchSection: some View {
        PassportSection(title: "Your match", subtitle: "Relative relevance, not a probability of eligibility.") {
            VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                HStack {
                    Text(RelevanceTier(score: result.score).label)
                        .font(.headline)
                        .foregroundStyle(Theme.Color.ink)
                    Spacer()
                    Label(eligibility.shortLabel, systemImage: eligibility.symbolName)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(eligibility.tintColor)
                }

                MatchTraceView(signals: signals)

                if !result.explanation.manual_review_signals.isEmpty {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        Text("Needs manual review")
                            .font(.sectionLabel)
                            .foregroundStyle(Theme.Color.muted)
                        ForEach(result.explanation.manual_review_signals, id: \.self) { note in
                            Label(note, systemImage: "person.fill.questionmark")
                                .font(.caption)
                                .foregroundStyle(Theme.Color.ink)
                        }
                    }
                }
            }
        }
    }

    // MARK: C. Trial overview

    private var overviewSection: some View {
        PassportSection(title: "Trial overview") {
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                if let summary = trial.brief_summary, !summary.isEmpty {
                    Text(summary)
                        .font(.body)
                        .foregroundStyle(Theme.Color.ink)
                }
                overviewRow(label: "Interventions", value: trial.interventions)
                overviewRow(label: "Phase", value: trial.phases)
                overviewRow(label: "Locations", value: trial.locations)
            }
        }
    }

    private func overviewRow(label: String, value: [String]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased())
                .font(.caption2.weight(.semibold))
                .foregroundStyle(Theme.Color.muted)
            Text(value.isEmpty ? "Not listed" : value.joined(separator: ", "))
                .font(.subheadline)
                .foregroundStyle(Theme.Color.ink)
        }
    }

    // MARK: D. Eligibility evidence

    private var eligibilityEvidenceSection: some View {
        PassportSection(title: "Eligibility evidence", subtitle: "Structured checks only -- not full medical eligibility.") {
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                overviewRow(label: "Age range", value: [
                    "Min \(trial.minimum_age ?? "not specified")",
                    "Max \(trial.maximum_age ?? "not specified")",
                ])
                overviewRow(label: "Sex requirement", value: [trial.sex ?? "Not specified"])
                if let criteria = trial.eligibility_criteria, !criteria.isEmpty {
                    DisclosureGroup("Full eligibility criteria text") {
                        Text(criteria)
                            .font(.footnote)
                            .foregroundStyle(Theme.Color.ink)
                            .padding(.top, Theme.Spacing.xs)
                    }
                    .font(.caption.weight(.medium))
                    .tint(Theme.Color.accent)
                }
                if !result.explanation.eligibility_notes.isEmpty {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        Text("Needs confirmation")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(Theme.Color.muted)
                        ForEach(result.explanation.eligibility_notes, id: \.self) { note in
                            Label(note, systemImage: "exclamationmark.circle")
                                .font(.caption)
                                .foregroundStyle(Theme.Color.ink)
                        }
                    }
                }
            }
        }
    }

    // MARK: E. Representation & Access

    private var representationSection: some View {
        PassportSection(title: "Representation & access") {
            ESRScoreView(esr: result.esr)
        }
    }

    // MARK: G. Source

    private var sourceSection: some View {
        PassportSection(title: "Source") {
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                if let urlString = trial.source_url, let url = URL(string: urlString) {
                    Link(destination: url) {
                        Label("View on ClinicalTrials.gov", systemImage: "arrow.up.right.square")
                            .font(.subheadline.weight(.medium))
                    }
                    .tint(Theme.Color.accent)
                }
                ProvenanceText(text: "Response source: \(responseSource)")
                if let sexSource = trial.enrollment_sex_source {
                    ProvenanceText(text: "Sex enrollment source: \(sexSource)")
                }
                if let raceSource = trial.enrollment_race_source {
                    ProvenanceText(text: "Race enrollment source: \(raceSource)")
                }
            }
        }
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
    }
}
