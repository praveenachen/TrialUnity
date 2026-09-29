import SwiftUI

/// One shortlist row: title, NCT ID, status, location, phase, a qualitative
/// relevance tier (never a bare score), a one-line eligibility read, and a
/// one-line ESR/evidence read. Deliberately not a dense multi-stat card.
struct TrialResultRow: View {
    let result: TrialRecommendation

    private var tier: RelevanceTier { RelevanceTier(score: result.score) }
    private var eligibility: EligibilitySummary { EligibilitySummary(status: result.structured_eligibility?.status) }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(result.trial.title)
                .font(.body.weight(.semibold))
                .foregroundStyle(Theme.Color.ink)
                .lineLimit(2)

            HStack(spacing: Theme.Spacing.xs) {
                ProvenanceText(text: result.trial.nct_id)
                Text("·").foregroundStyle(Theme.Color.muted)
                Text(result.trial.status.capitalized)
                    .font(.caption)
                    .foregroundStyle(Theme.Color.muted)
                if let phase = result.trial.phases.first {
                    Text("·").foregroundStyle(Theme.Color.muted)
                    Text(phase.replacingOccurrences(of: "_", with: " ").capitalized)
                        .font(.caption)
                        .foregroundStyle(Theme.Color.muted)
                }
            }

            if let location = result.trial.locations.first {
                Label(location, systemImage: "mappin.and.ellipse")
                    .font(.caption)
                    .foregroundStyle(Theme.Color.muted)
                    .lineLimit(1)
            }

            HStack(spacing: Theme.Spacing.s) {
                StatusPill(text: tier.label, symbolName: tier.symbolName, tint: tier.tintColor)
                StatusPill(text: eligibility.shortLabel, symbolName: eligibility.symbolName, tint: eligibility.tintColor)
            }

            HStack(spacing: Theme.Spacing.xs) {
                Image(systemName: "shield.checkerboard")
                    .font(.caption)
                    .foregroundStyle(Theme.Color.muted)
                Text(evidenceSummary)
                    .font(.caption)
                    .foregroundStyle(Theme.Color.muted)
            }
        }
        .padding(.vertical, Theme.Spacing.s)
        .accessibilityElement(children: .combine)
    }

    private var evidenceSummary: String {
        guard let esr = result.esr else { return "Access evidence unavailable" }
        guard let score = esr.score, score.isFinite else { return "Access evidence: limited" }
        return "Access evidence: \(ScoreFormat.rounded(score))/100"
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
    }
}
