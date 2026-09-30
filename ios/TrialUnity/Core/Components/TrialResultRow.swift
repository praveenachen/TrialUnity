import SwiftUI

/// One shortlist row, styled as a restrained bordered card: identity line
/// (NCT ID / status / phase), title, location, then a compact footer of
/// qualitative tiers (never a bare score) and a one-line ESR/evidence read.
/// Deliberately not a dense multi-stat card, and deliberately not a second
/// copy of the Trial Passport.
struct TrialResultRow: View {
    @Environment(SavedTrialsStore.self) private var savedTrials
    let result: TrialRecommendation
    var preferredLocation: String? = nil

    private var tier: RelevanceTier { RelevanceTier(score: result.score) }
    private var eligibility: EligibilitySummary { EligibilitySummary(status: result.structured_eligibility?.status) }

    var body: some View {
        CardContainer(padding: Theme.Spacing.m) {
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                HStack(alignment: .top) {
                    HStack(spacing: Theme.Spacing.xs) {
                        ProvenanceText(text: result.trial.nct_id)
                        Text("·").foregroundStyle(Theme.Color.muted)
                        Text(result.trial.status.capitalized)
                            .font(.caption.weight(.medium))
                            .foregroundStyle(result.trial.status.uppercased() == "RECRUITING" ? Theme.Color.evidence : Theme.Color.muted)
                        if let phase = result.trial.phases.first {
                            Text("·").foregroundStyle(Theme.Color.muted)
                            Text(phase.replacingOccurrences(of: "_", with: " ").capitalized)
                                .font(.caption)
                                .foregroundStyle(Theme.Color.muted)
                        }
                    }
                    Spacer(minLength: Theme.Spacing.s)
                    if savedTrials.contains(result.id) {
                        Image(systemName: "bookmark.fill")
                            .font(.caption)
                            .foregroundStyle(Theme.Color.accent)
                            .accessibilityLabel("Saved")
                    }
                }

                Text(result.trial.title)
                    .font(.editorialHeadline)
                    .foregroundStyle(Theme.Color.ink)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)

                if let location = PatientPresentation.location(result.trial, near: preferredLocation) {
                    Label(location, systemImage: "mappin.and.ellipse")
                        .font(.caption)
                        .foregroundStyle(Theme.Color.muted)
                        .lineLimit(1)
                }

                FlowLayout(spacing: Theme.Spacing.s) {
                    StatusPill(text: tier.label, symbolName: tier.symbolName, tint: tier.tintColor)
                    StatusPill(text: eligibility.shortLabel, symbolName: eligibility.symbolName, tint: eligibility.tintColor)
                }

                Text(evidenceSummary)
                    .font(.caption)
                    .foregroundStyle(Theme.Color.muted)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var evidenceSummary: String {
        guard let esr = result.esr else { return "Access evidence unavailable" }
        guard let score = esr.score, score.isFinite else { return "Access evidence: limited" }
        let coverage = esr.evidence_coverage.isFinite ? ScoreFormat.clamped(esr.evidence_coverage) : 0
        return "Representation & access · \(ScoreFormat.rounded(score)) · \(PatientPresentation.evidence(coverage))"
    }
}

#Preview {
    List {
        TrialResultRow(result: TrialRecommendation(
            trial: Trial(
                nct_id: "NCT00000001", title: "Phase 2 Study of Trastuzumab Deruxtecan in Metastatic Breast Cancer",
                status: "RECRUITING", conditions: ["Metastatic Breast Cancer"], interventions: [], phases: ["PHASE2"],
                brief_summary: nil, eligibility_criteria: nil, sex: "FEMALE", minimum_age: "18 Years", maximum_age: nil,
                locations: ["Boston, Massachusetts, United States"], sponsor: nil, source_url: nil,
                enrollment_sex_distribution: nil, enrollment_race_distribution: nil,
                enrollment_sex_source: nil, enrollment_race_source: nil, target_enrollment: nil
            ),
            score: 0.72, score_breakdown: [:],
            relevance: RelevanceScores(overall: 0.72, lexical: 0.6, semantic: 0.7, structured: [:], weighted_components: [:], weights: [:]),
            explanation: MatchExplanation(matched_terms: [], eligibility_notes: [], ranking_rationale: "", patient_friendly_summary: "", relevant_signals: [], manual_review_signals: []),
            structured_eligibility: StructuredEligibility(status: "compatible", criteria: [:]),
            esr: ESRResult(score: 63, mode: "mixed", evidence_coverage: 0.6, components: [:], weights_used: [:]),
            representation_risk: nil
        ))
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
    }
    .environment(SavedTrialsStore())
}
