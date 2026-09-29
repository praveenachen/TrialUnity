import XCTest
@testable import TrialUnityIntegration

final class PresentationLogicTests: XCTestCase {
    // MARK: RelevanceTier

    func testRelevanceTierThresholds() {
        XCTAssertEqual(RelevanceTier(score: 0.9), .strong)
        XCTAssertEqual(RelevanceTier(score: 0.6), .strong)
        XCTAssertEqual(RelevanceTier(score: 0.59), .possible)
        XCTAssertEqual(RelevanceTier(score: 0.3), .possible)
        XCTAssertEqual(RelevanceTier(score: 0.29), .needsReview)
        XCTAssertEqual(RelevanceTier(score: 0), .needsReview)
    }

    // MARK: EligibilitySummary

    func testEligibilitySummaryNeverTreatsMissingStatusAsCompatible() {
        XCTAssertEqual(EligibilitySummary(status: "compatible"), .compatible)
        XCTAssertEqual(EligibilitySummary(status: "incompatible"), .incompatible)
        XCTAssertEqual(EligibilitySummary(status: "unknown"), .unknown)
        XCTAssertEqual(EligibilitySummary(status: nil), .unknown)
        XCTAssertEqual(EligibilitySummary(status: "garbage"), .unknown)
    }

    // MARK: ESRDisplay

    func testESRDisplayOrdersKnownComponentsFirstThenSortsExtras() {
        let components: [String: ComponentEvidence] = [
            "race": ComponentEvidence(score: nil, evidence_coverage: 0, evidence_type: "insufficient_data", rationale: "", source: nil, missing_evidence: []),
            "zzz_extra": ComponentEvidence(score: 1, evidence_coverage: 1, evidence_type: "observed_enrollment", rationale: "", source: nil, missing_evidence: []),
            "sex": ComponentEvidence(score: 100, evidence_coverage: 1, evidence_type: "protocol_inclusivity", rationale: "", source: nil, missing_evidence: []),
            "socioeconomic": ComponentEvidence(score: 36, evidence_coverage: 1, evidence_type: "observed_geographic", rationale: "", source: nil, missing_evidence: []),
        ]
        let ordered = ESRDisplay.orderedComponents(components).map(\.key)
        XCTAssertEqual(ordered, ["socioeconomic", "sex", "race", "zzz_extra"])
    }

    func testESRDisplayLabels() {
        XCTAssertEqual(ESRDisplay.label(for: "socioeconomic"), "Socioeconomic access")
        XCTAssertEqual(ESRDisplay.label(for: "sex"), "Sex inclusivity")
        XCTAssertEqual(ESRDisplay.label(for: "race"), "Race representation")
    }

    // MARK: ESREvidenceType

    func testESREvidenceTypeNeverLabelsMissingDataAsObserved() {
        XCTAssertEqual(ESREvidenceType(rawValue: "insufficient_data").label, "Data unavailable")
        XCTAssertEqual(ESREvidenceType(rawValue: "insufficient_benchmark").label, "Benchmark unavailable")
        XCTAssertEqual(ESREvidenceType(rawValue: "observed_geographic").label, "Observed evidence")
        XCTAssertEqual(ESREvidenceType(rawValue: "protocol_inclusivity").label, "Prospective evidence")
    }

    // MARK: RiskLevelDisplay

    func testRiskLevelDisplayHandlesMissingLevel() {
        XCTAssertEqual(RiskLevelDisplay.label(nil), "Unavailable")
        XCTAssertEqual(RiskLevelDisplay.label("low"), "Low")
    }

    func testShortDriverStripsImportanceSuffix() {
        XCTAssertEqual(RiskLevelDisplay.shortDriver("num_sites (importance 0.552)"), "num_sites")
        XCTAssertEqual(RiskLevelDisplay.shortDriver("no_suffix"), "no_suffix")
    }

    func testDriverLabelMapsKnownKeysToPatientFriendlyText() {
        XCTAssertEqual(RiskLevelDisplay.driverLabel("num_sites (importance 0.552)"), "Number of trial sites")
        XCTAssertEqual(RiskLevelDisplay.driverLabel("num_regions (importance 0.465)"), "Geographic reach")
        XCTAssertEqual(RiskLevelDisplay.driverLabel("target_enrollment_log (importance 0.301)"), "Target enrollment size")
    }

    func testDriverLabelHumanizesUnknownKeysRatherThanShowingRawSnakeCase() {
        XCTAssertEqual(RiskLevelDisplay.driverLabel("eligibility_criteria_line_count (importance 0.1)"), "Eligibility Criteria Line Count")
    }

    func testDriverLabelNeverMutatesTheRawKeyExtraction() {
        // driverLabel must not change what shortDriver (the raw-key extractor) returns.
        let raw = "num_sites (importance 0.552)"
        XCTAssertEqual(RiskLevelDisplay.shortDriver(raw), "num_sites")
        _ = RiskLevelDisplay.driverLabel(raw)
        XCTAssertEqual(RiskLevelDisplay.shortDriver(raw), "num_sites")
    }

    // MARK: ScoreFormat

    func testScoreFormatRoundedHandlesNilAndNonFiniteAsUnavailable() {
        XCTAssertEqual(ScoreFormat.rounded(63.4), "63")
        XCTAssertEqual(ScoreFormat.rounded(nil), "—")
        XCTAssertEqual(ScoreFormat.rounded(.nan), "—")
        XCTAssertEqual(ScoreFormat.rounded(.infinity), "—")
        XCTAssertEqual(ScoreFormat.rounded(-.infinity), "—")
    }

    func testScoreFormatClampedGuardsNaNAndInfinityAndRange() {
        XCTAssertEqual(ScoreFormat.clamped(0.5), 0.5)
        XCTAssertEqual(ScoreFormat.clamped(-1), 0)
        XCTAssertEqual(ScoreFormat.clamped(5), 1)
        // Non-finite values (NaN or either infinity) all fall back to the range's
        // lower bound -- matching CoverageBar's existing "non-finite -> 0" guard,
        // never left as a value that could still reach a CGFloat/frame calculation.
        XCTAssertEqual(ScoreFormat.clamped(.nan), 0)
        XCTAssertEqual(ScoreFormat.clamped(.infinity), 0)
        XCTAssertEqual(ScoreFormat.clamped(-.infinity), 0)
    }

    // MARK: MatchTraceBuilder

    private func makeTrial(
        conditions: [String] = ["Lung Cancer"], interventions: [String] = [], phases: [String] = [],
        sex: String? = nil, minimumAge: String? = nil, maximumAge: String? = nil, locations: [String] = []
    ) -> Trial {
        Trial(
            nct_id: "NCT1", title: "Trial", status: "RECRUITING", conditions: conditions,
            interventions: interventions, phases: phases, brief_summary: nil, eligibility_criteria: nil,
            sex: sex, minimum_age: minimumAge, maximum_age: maximumAge, locations: locations, sponsor: nil,
            source_url: nil, enrollment_sex_distribution: nil, enrollment_race_distribution: nil,
            enrollment_sex_source: nil, enrollment_race_source: nil, target_enrollment: nil
        )
    }

    private func makeResult(
        trial: Trial, structured: [String: Double] = [:], criteria: [String: EligibilityCriterion] = [:]
    ) -> TrialRecommendation {
        TrialRecommendation(
            trial: trial, score: 0.5, score_breakdown: [:],
            relevance: RelevanceScores(overall: 0.5, lexical: 0.5, semantic: 0.5, structured: structured, weighted_components: [:], weights: [:]),
            explanation: MatchExplanation(matched_terms: [], eligibility_notes: [], ranking_rationale: "", patient_friendly_summary: "", relevant_signals: [], manual_review_signals: []),
            structured_eligibility: StructuredEligibility(status: "unknown", criteria: criteria),
            esr: nil, representation_risk: nil
        )
    }

    func testMatchTraceTreatsUnspecifiedPatientPreferencesAsUnknownNotNoMatch() {
        let profile = PatientProfile(age: nil, sex: nil, condition: "lung cancer", location: nil, intervention_preferences: [], phase_preferences: [], notes: nil)
        let result = makeResult(trial: makeTrial(), structured: ["intervention": 0, "location": 0, "phase": 0])
        let signals = MatchTraceBuilder.build(profile: profile, result: result)

        let treatment = signals.first { $0.id == "treatment" }
        let location = signals.first { $0.id == "location" }
        let phase = signals.first { $0.id == "phase" }
        XCTAssertEqual(treatment?.status, .unknown)
        XCTAssertEqual(location?.status, .unknown)
        XCTAssertEqual(phase?.status, .unknown)
    }

    func testMatchTraceAgeSignalIsIncompatibleWhenEitherBoundFails() {
        let profile = PatientProfile(age: 10, sex: nil, condition: "cancer", location: nil, intervention_preferences: [], phase_preferences: [], notes: nil)
        let result = makeResult(trial: makeTrial(minimumAge: "18 Years"), criteria: [
            "minimum_age": EligibilityCriterion(state: "incompatible", reason: "Patient age 10 years; trial minimum age 18 years."),
            "maximum_age": EligibilityCriterion(state: "compatible", reason: "Trial explicitly specifies no maximum age limit."),
        ])
        let signals = MatchTraceBuilder.build(profile: profile, result: result)
        XCTAssertEqual(signals.first { $0.id == "age" }?.status, .noMatch)
    }

    func testMatchTraceConditionSignalReflectsStructuredScore() {
        let profile = PatientProfile(age: nil, sex: nil, condition: "lung cancer", location: nil, intervention_preferences: [], phase_preferences: [], notes: nil)
        let matched = makeResult(trial: makeTrial(), structured: ["condition": 1])
        let unmatched = makeResult(trial: makeTrial(), structured: ["condition": 0])
        XCTAssertEqual(MatchTraceBuilder.build(profile: profile, result: matched).first { $0.id == "condition" }?.status, .match)
        XCTAssertEqual(MatchTraceBuilder.build(profile: profile, result: unmatched).first { $0.id == "condition" }?.status, .noMatch)
    }

    func testMatchTraceBuildsExactlySixSignalsInOrder() {
        let profile = PatientProfile(age: nil, sex: nil, condition: "cancer", location: nil, intervention_preferences: [], phase_preferences: [], notes: nil)
        let signals = MatchTraceBuilder.build(profile: profile, result: makeResult(trial: makeTrial()))
        XCTAssertEqual(signals.map(\.id), ["condition", "age", "sex", "treatment", "location", "phase"])
    }
}
