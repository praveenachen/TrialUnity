import XCTest
@testable import TrialUnityIntegration

final class DocumentProfileAssistTests: XCTestCase {
    private func extract(_ text: String, confidence: Float = 0.99) -> [ScanCandidate] {
        DocumentProfileAssist.extract(text.components(separatedBy: "\n").map { ScanTextLine(text: $0, confidence: confidence) }, now: ISO8601DateFormatter().date(from: "2026-10-01T12:00:00Z")!)
    }
    func testDiagnosisTreatmentAndBiomarkerMentions() {
        let result = extract("Diagnosis: Non-small cell lung cancer\nPrior treatment: Pembrolizumab\nPD-L1 negative")
        XCTAssertEqual(result.filter { $0.field == .condition }.map(\.value), ["Non-small cell lung cancer"])
        XCTAssertEqual(result.filter { $0.field == .treatment }.map(\.value), ["Pembrolizumab"])
        XCTAssertEqual(result.filter { $0.field == .biomarker }.map(\.value), ["PD-L1 negative"])
        XCTAssertTrue(result.first { $0.field == .biomarker }!.needsReview)
    }
    func testAgeAndUnambiguousDOB() {
        XCTAssertEqual(extract("Age: 54").first?.value, "54")
        XCTAssertEqual(extract("54-year-old patient").first?.value, "54")
        XCTAssertEqual(extract("DOB: 1972-10-02").first?.value, "53")
        XCTAssertTrue(extract("DOB: 1972-10-02").first!.needsReview)
        for value in ["DOB: 04/05/1972", "DOB: 2027-01-01", "DOB: 1972-02-31", "Age: 130", "Age: 54 months", "Record: 54"] {
            XCTAssertTrue(extract(value).isEmpty, value)
        }
    }
    func testAmbiguousNegatedAndNoisyText() {
        XCTAssertTrue(extract("Possible diagnosis: lung cancer\nFamily history: diagnosed with breast cancer\nNo evidence of diagnosis: melanoma\nDiagnosis: no lung cancer\nTreatment: planned surgery").isEmpty)
        XCTAssertTrue(extract("Diagnosis: breast cancer", confidence: 0.5).first!.needsReview)
        let conflicts = extract("Age: 54\nAge: 58")
        XCTAssertEqual(conflicts.count, 2)
        XCTAssertTrue(conflicts.allSatisfy(\.needsReview))
        XCTAssertTrue(extract("Unclear handwritten report").isEmpty)
    }
    func testPendingSuggestionsDoNotAffectPayloadAndAcceptanceUsesExistingFields() throws {
        let draft = PatientProfileDraft()
        draft.condition = "Manual diagnosis"; draft.ageText = "60"
        draft.notes = "Manual notes"; draft.interventionPreferences = ["Surgery"]
        let baseline = PatientProfile(draft: draft)
        let candidates = extract("Diagnosis: breast cancer\nAge: 54\nTreatment: Pembrolizumab\nPD-L1 positive, TPS 60%")
        draft.keepScanSuggestions(candidates)
        let pending = PatientProfile(draft: draft)
        XCTAssertEqual(pending.condition, baseline.condition)
        XCTAssertEqual(pending.age, baseline.age)
        XCTAssertEqual(pending.notes, baseline.notes)
        XCTAssertEqual(pending.intervention_preferences, baseline.intervention_preferences)
        XCTAssertEqual(draft.scanSuggestions.count, 4)
        for item in candidates where item.field == .condition || item.field == .age {
            XCTAssertFalse(draft.useScanSuggestion(item.id))
            XCTAssertFalse(draft.acceptedScanSuggestionIDs.contains(item.id))
            XCTAssertTrue(draft.useScanSuggestion(item.id, replaceExisting: true))
        }
        let treatment = try XCTUnwrap(candidates.first { $0.field == .treatment })
        XCTAssertTrue(draft.useScanSuggestion(treatment.id))
        XCTAssertEqual(draft.notes, "Manual notes")
        let biomarker = try XCTUnwrap(candidates.first { $0.field == .biomarker })
        XCTAssertEqual(biomarker.value, "PD-L1 positive, TPS 60%")
        XCTAssertTrue(draft.useScanSuggestion(biomarker.id))
        let result = PatientProfile(draft: draft)
        XCTAssertEqual(result.condition, "Breast cancer")
        XCTAssertEqual(result.age, 54)
        XCTAssertEqual(result.intervention_preferences, ["Surgery", "Pembrolizumab"])
        XCTAssertEqual(result.notes, "Manual notes\nBiomarkers: PD-L1 positive, TPS 60%")
        XCTAssertEqual(draft.acceptedScanSuggestionIDs.count, 4)
        XCTAssertFalse(draft.useScanSuggestion(treatment.id))
    }
    func testNewScanUpdatesPendingWithoutReplacingConfirmedValues() throws {
        let draft = PatientProfileDraft()
        draft.keepScanSuggestions(extract("Diagnosis: breast cancer\nAge: 54"))
        let condition = try XCTUnwrap(draft.scanSuggestions.first { $0.field == .condition })
        XCTAssertTrue(draft.useScanSuggestion(condition.id))
        draft.keepScanSuggestions(extract("Diagnosis: lung cancer\nAge: 60"))
        XCTAssertEqual(draft.condition, "Breast cancer")
        XCTAssertTrue(draft.ageText.isEmpty)
        XCTAssertTrue(draft.acceptedScanSuggestionIDs.contains(condition.id))
        XCTAssertFalse(draft.scanSuggestions.contains { $0.field == .age && $0.value == "54" })
        XCTAssertTrue(draft.scanSuggestions.contains { $0.value == "Lung cancer" })
        XCTAssertNil(PatientProfile(draft: draft).age)
        XCTAssertNil(PatientProfile(draft: draft).notes)
    }
    func testRawTextAndUnacceptedValuesAreAbsentFromPayload() throws {
        let draft = PatientProfileDraft()
        draft.keepScanSuggestions(extract("Identifier SECRET-123\nDiagnosis: breast cancer\nTreatment: Pembrolizumab"))
        let data = try JSONEncoder().encode(PatientProfile(draft: draft))
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertFalse(json.contains("SECRET"))
        XCTAssertFalse(json.contains("Pembrolizumab"))
        XCTAssertFalse(json.contains("Breast cancer"))
        XCTAssertTrue(draft.notes.isEmpty)
    }
    func testEquivalentLabelsAndTermsDeduplicate() {
        let result = extract("Dx: NSCLC\nDiagnosis: non small cell lung carcinoma\nCondition: Non-small cell lung cancer\nMeds: chemo\nTreatment: Chemotherapy\nPDL1\nPD-L1 positive, TPS 60%\nPD L1 positive, TPS 60%")
        XCTAssertEqual(result.filter { $0.field == .condition }.map(\.value), ["Non-small cell lung cancer"])
        XCTAssertEqual(result.filter { $0.field == .treatment }.map(\.value), ["Chemotherapy"])
        XCTAssertEqual(result.filter { $0.field == .biomarker }.map(\.value), ["PD-L1 positive, TPS 60%"])
    }
    func testAgeVariantsAndSplitOCRLabels() {
        for text in ["Age 54", "Patient age = 54 years", "Age is 54", "Age (years): 54", "Aged 54", "54 y/o", "54 yo", "54 years old", "54-yr-old", "Age:\n54"] {
            XCTAssertEqual(extract(text).filter { $0.field == .age }.map(\.value), ["54"], text)
        }
        XCTAssertEqual(extract("Diagnosis:\nNSCLC").first?.value, "Non-small cell lung cancer")
        XCTAssertEqual(extract("Birth date 1972-10-02").first?.value, "53")
        XCTAssertTrue(extract("Age: 18 months\nAge: 2 weeks\nAge: 4 days").isEmpty)
        XCTAssertTrue(extract("Family history: aged 54, Dx NSCLC").isEmpty)
        XCTAssertEqual(extract("Age 54\n54 years old\nAge:\n54").count, 1)
    }
    func testConflictingBiomarkerResultsAreNotMerged() {
        let result = extract("PD-L1 positive\nPDL1 negative\nPD-L1")
        XCTAssertEqual(result.map(\.value), ["PD-L1 positive", "PD-L1 negative"])
        XCTAssertTrue(result.allSatisfy(\.needsReview))
    }

    func testColumnWiseTableOCRPairsLabelsWithValues() {
        // Vision can return a table as all labels, then all values.
        let result = extract("Patient\nAge\nDiagnosis\nCurrent treatment\nBiomarker\nVisit date\nJamie Lee (fictional)\n54\nNon-small cell lung cancer (NSCLC)\nPembrolizumab\nPD-L1 positive, TPS 60%\nSeptember 18, 2026")
        XCTAssertEqual(result.filter { $0.field == .age }.map(\.value), ["54"])
        XCTAssertEqual(result.filter { $0.field == .condition }.map(\.value), ["Non-small cell lung cancer"])
        XCTAssertEqual(result.filter { $0.field == .treatment }.map(\.value), ["Pembrolizumab"])
        XCTAssertEqual(result.filter { $0.field == .biomarker }.map(\.value), ["PD-L1 positive, TPS 60%"])
        XCTAssertTrue(result.filter { $0.field == .age || $0.field == .condition }.allSatisfy(\.needsReview))
    }

    func testRowWiseTableStillExtracts() {
        let result = extract("Age 54\nDiagnosis Non-small cell lung cancer (NSCLC)")
        XCTAssertEqual(result.filter { $0.field == .age }.map(\.value), ["54"])
        XCTAssertEqual(result.filter { $0.field == .condition }.map(\.value), ["Non-small cell lung cancer"])
    }

    /// Lines carry page geometry, so pairing must not depend on the order Vision returned them.
    func testGeometryPairsLabelsWithValuesInAnyOrder() {
        func line(_ text: String, x: CGFloat, row: Int, width: CGFloat = 0.3) -> ScanTextLine {
            ScanTextLine(text: text, confidence: 0.99, box: CGRect(x: x, y: 0.9 - CGFloat(row) * 0.06, width: width, height: 0.03))
        }
        let rows = [("Patient", "Jamie Lee (fictional)"), ("Age", "54"), ("Diagnosis", "Non-small cell lung cancer (NSCLC)"),
                    ("Current treatment", "Pembrolizumab"), ("Biomarker", "PD-L1 positive, TPS 60%")]
        var lines: [ScanTextLine] = []
        for (row, pair) in rows.enumerated() { lines += [line(pair.0, x: 0.1, row: row), line(pair.1, x: 0.4, row: row)] }
        for ordering in [lines, lines.reversed(), lines.shuffled()] {
            let result = DocumentProfileAssist.extract(Array(ordering))
            XCTAssertEqual(result.filter { $0.field == .age }.map(\.value), ["54"])
            XCTAssertEqual(result.filter { $0.field == .condition }.map(\.value), ["Non-small cell lung cancer"])
            XCTAssertEqual(result.filter { $0.field == .treatment }.map(\.value), ["Pembrolizumab"])
        }
    }

    func testGeometryPairsValueBelowLabel() {
        let lines = [
            ScanTextLine(text: "54", confidence: 0.99, box: CGRect(x: 0.1, y: 0.80, width: 0.1, height: 0.03)),
            ScanTextLine(text: "Age", confidence: 0.99, box: CGRect(x: 0.1, y: 0.85, width: 0.1, height: 0.03)),
        ]
        XCTAssertEqual(DocumentProfileAssist.extract(lines).filter { $0.field == .age }.map(\.value), ["54"])
    }
}
