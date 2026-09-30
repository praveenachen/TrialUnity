import XCTest
@testable import TrialUnityIntegration

final class CompositionTests: XCTestCase {
    func testVisualBoundaries() {
        for value in [Double.nan, .infinity, -.infinity, -1, 0, 1, Double.greatestFiniteMagnitude] {
            XCTAssertTrue(VisualNumber.dimension(value).isFinite)
            XCTAssertGreaterThanOrEqual(VisualNumber.dimension(value), 0)
            XCTAssertTrue(VisualNumber.fraction(value, 0).isFinite)
            XCTAssertTrue(VisualNumber.fraction(value, .infinity).isFinite)
            XCTAssertTrue(ScoreFormat.clamped(value).isFinite)
            _ = ScoreFormat.rounded(value)
        }
        XCTAssertEqual(VisualNumber.fraction(1, 0), 0)
        XCTAssertEqual(VisualNumber.fraction(2, 1), 1)
    }
    func testEvidenceLabelsAreNotScores() {
        XCTAssertEqual(PatientPresentation.evidence(0.6), "Moderate evidence")
        XCTAssertEqual(PatientPresentation.evidence(.nan), "Limited evidence")
        XCTAssertEqual(ScoreFormat.rounded(nil), "—")
    }
    func testSummaryUsesOnlyOriginalWords() {
        let source = "First sentence. Second sentence. Third sentence."
        XCTAssertEqual(PatientPresentation.summary(source), "First sentence. Second sentence")
        XCTAssertEqual(PatientPresentation.summary(nil), "Study summary not reported.")
    }
}
