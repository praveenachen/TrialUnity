import Foundation

struct ScanCandidate: Identifiable, Equatable {
    enum Field: String { case condition = "Condition", treatment = "Treatment mentioned", biomarker = "Biomarker mentioned", age = "Age" }
    let field: Field
    let value: String
    let needsReview: Bool
    var id: String { field.rawValue + ":" + value.lowercased() }
}

struct ScanTextLine {
    let text: String
    let confidence: Float
    /// Normalized page rect from Vision (origin bottom-left). When present, labels are
    /// paired with values by position on the page instead of by OCR reading order.
    var box: CGRect? = nil
    var page = 0
}

enum DocumentProfileAssist {
    static func extract(_ lines: [ScanTextLine], now: Date = Date()) -> [ScanCandidate] {
        var results: [ScanCandidate] = []
        func add(_ field: ScanCandidate.Field, _ value: String, _ review: Bool) {
            let candidate = ScanCandidate(field: field, value: value, needsReview: review)
            if let existing = results.firstIndex(where: { $0.id == candidate.id }) {
                if review && !results[existing].needsReview { results[existing] = candidate }
            } else { results.append(candidate) }
        }
        // Join standalone field labels to their value lines (geometry first, reading order as fallback).
        let labelPattern = #"^\s*(?:diagnosis|diagnoses|dx|condition|primary diagnosis|medical condition|age|patient age|age in years|dob|d\.o\.b\.|date of birth|birth date|medications?|treatments?|current (?:treatment|medications?|therapy)|biomarkers?|patient|patient name|name|sex|gender|visit date|date of visit)\s*[:=-]?\s*$"#
        func isLabel(_ i: Int) -> Bool { matches(labelPattern, lines[i].text) }
        func joined(_ label: ScanTextLine, _ value: ScanTextLine, _ confidence: Float) -> ScanTextLine {
            ScanTextLine(text: label.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ":=-"))) + ": " + value.text,
                         confidence: confidence)
        }
        // Position-based pairing: the value is the nearest non-label line to the right on the
        // same row, else the nearest line directly below. Independent of OCR reading order.
        func partner(of i: Int) -> Int? {
            guard let lb = lines[i].box else { return nil }
            let others = lines.indices.filter { $0 != i && lines[$0].page == lines[i].page && lines[$0].box != nil && !isLabel($0) }
            let right = others.filter { j in
                let b = lines[j].box!
                let overlap = min(lb.maxY, b.maxY) - max(lb.minY, b.minY)
                return b.minX >= lb.midX && overlap > 0.5 * min(lb.height, b.height)
            }
            if let j = right.min(by: { lines[$0].box!.minX < lines[$1].box!.minX }) { return j }
            let below = others.filter { j in
                let b = lines[j].box!
                let horizontal = min(lb.maxX, b.maxX) - max(lb.minX, b.minX)
                let gap = lb.minY - b.maxY
                return horizontal > 0 && gap > -0.2 * lb.height && gap < 3 * lb.height
            }
            return below.min(by: { lb.minY - lines[$0].box!.maxY < lb.minY - lines[$1].box!.maxY })
        }
        var prepared: [ScanTextLine] = []
        if lines.contains(where: { $0.box != nil }) {
            var consumed = Set<Int>()
            var pairs: [Int: Int] = [:]
            for i in lines.indices where isLabel(i) {
                if let j = partner(of: i), !consumed.contains(j) { pairs[i] = j; consumed.insert(j) }
            }
            for i in lines.indices where !consumed.contains(i) {
                if let j = pairs[i] { prepared.append(joined(lines[i], lines[j], min(lines[i].confidence, lines[j].confidence))) }
                else { prepared.append(lines[i]) }
            }
        } else {
            // No geometry (plain text): fall back to reading order. A run of N labels followed
            // by N values is a column-wise table; a lone label takes the next line.
            var index = 0
        while index < lines.count {
            guard isLabel(index), index + 1 < lines.count else { prepared.append(lines[index]); index += 1; continue }
            var run = 1
            while index + run < lines.count, isLabel(index + run) { run += 1 }
            let valuesStart = index + run
            if run >= 2, valuesStart + run <= lines.count {
                // Column-wise table: pair by position, and flag for review since the layout was inferred.
                for offset in 0..<run {
                    let label = lines[index + offset], value = lines[valuesStart + offset]
                    prepared.append(joined(label, value, min(label.confidence, value.confidence, 0.8)))
                }
                index = valuesStart + run
            } else {
                prepared.append(joined(lines[index], lines[index + 1], min(lines[index].confidence, lines[index + 1].confidence)))
                index += 2
            }
            }
        }
        for line in prepared {
            let text = line.text.replacingOccurrences(of: "–", with: "-")
                .replacingOccurrences(of: "—", with: "-")
                .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            // Negated, hypothetical, and family-history lines cannot establish patient facts.
            let ambiguous = matches(#"\b(possible|suspected|consider|rule out|ruled out|family history|mother|father|denies|negative|no|without|not|planned|recommended)\b"#, text)
            let review = line.confidence < 0.85 || ambiguous
            if matches(#"\b(diagnosis|diagnoses|dx|diagnosed with|condition|problem list|medical history)\s*:?"#, text), !ambiguous {
                let groups: [(String, String)] = [
                    ("Non-small cell lung cancer", #"\b(?:non[- ]small[- ]cell lung (?:cancer|carcinoma)|NSCLC)\b"#),
                    ("Small cell lung cancer", #"(?<!non-)(?<!non )\b(?:small[- ]cell lung (?:cancer|carcinoma)|SCLC)\b"#),
                    ("Breast cancer", #"\bbreast (?:cancer|carcinoma)\b"#),
                    ("Lung cancer", #"\blung (?:cancer|carcinoma)\b"#),
                    ("Colorectal cancer", #"\bcolorectal (?:cancer|carcinoma)\b"#),
                    ("Prostate cancer", #"\bprostate (?:cancer|carcinoma)\b"#),
                    ("Melanoma", #"\bmelanoma\b"#),
                    ("Type 2 diabetes", #"\b(?:type[- ]2 diabetes(?: mellitus)?|T2DM)\b"#)
                ]
                for (name, pattern) in groups where matches(pattern, text) {
                    if name == "Lung cancer", groups.prefix(2).contains(where: { matches($0.1, text) }) { continue }
                    add(.condition, name, review)
                }
            }
            if matches(#"\b(treatments?|therapies|therapy|medications?|meds|regimen|received|taking|treated|current|prior|previous)\b"#, text), !ambiguous {
                for name in ["Pembrolizumab", "Nivolumab", "Osimertinib", "Trastuzumab", "Cisplatin", "Carboplatin", "Paclitaxel", "Docetaxel", "Chemotherapy", "Immunotherapy", "Radiation therapy", "Surgery"] {
                    let pattern = name == "Chemotherapy" ? #"\b(?:chemotherapy|chemo)\b"#
                        : name == "Radiation therapy" ? #"\b(?:radiation therapy|radiotherapy)\b"#
                        : "\\b" + NSRegularExpression.escapedPattern(for: name) + "\\b"
                    if matches(pattern, text) { add(.treatment, name, true) }
                }
            }
            for name in ["PD-L1", "EGFR", "ALK", "HER2", "BRCA1", "BRCA2", "KRAS", "BRAF", "MSI", "MMR"] {
                let marker = name == "PD-L1" ? #"PD[- ]?L1"# : name == "HER2" ? #"HER[- ]?2"# : NSRegularExpression.escapedPattern(for: name)
                if matches("\\b" + marker + "\\b", text) {
                    var value = name
                    if let status = capture(marker + #"\s*[:=-]?\s*(positive|negative|detected|not detected)\b"#, text) {
                        value += " " + status.lowercased()
                    }
                    if name == "PD-L1", let tps = capture(#"\bTPS\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%"#, text),
                       let number = Double(tps), (0...100).contains(number) { value += ", TPS " + tps + "%" }
                    add(.biomarker, value, true)
                }
            }
            if !ambiguous {
                for pattern in [#"\b(?:patient\s+)?age(?:\s+in\s+years|\s*\(years\))?\s*(?:[:=,-]|is)?\s*(\d{1,3})\b(?!\s*(?:months?|weeks?|days?)\b)"#,
                    #"\baged\s+(\d{1,3})\b(?!\s*(?:months?|weeks?|days?)\b)"#,
                    #"\b(\d{1,3})[- ](?:years?[- ]old|yrs?[- ]old|y/o|yo)\b"#] {
                    if let number = capture(pattern, text), let age = Int(number), (0...120).contains(age) { add(.age, String(age), review) }
                }
                // Accept only unambiguous ISO DOBs; ambiguous numeric dates are deliberately omitted.
                if let dob = capture(#"\b(?:DOB|D\.O\.B\.|date of birth|birth date)\s*[:=-]?\s*(\d{4}-\d{2}-\d{2})\b"#, text) {
                    let formatter = DateFormatter()
                    formatter.locale = Locale(identifier: "en_US_POSIX")
                    formatter.timeZone = TimeZone(secondsFromGMT: 0)
                    formatter.dateFormat = "yyyy-MM-dd"
                    formatter.isLenient = false
                    if let date = formatter.date(from: dob), formatter.string(from: date) == dob, date <= now {
                        var calendar = Calendar(identifier: .gregorian)
                        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
                        if let age = calendar.dateComponents([.year], from: date, to: now).year, (0...120).contains(age) { add(.age, String(age), true) }
                    }
                }
            }
        }
        let allResults = results
        results.removeAll { candidate in
            candidate.field == .biomarker && allResults.contains { other in
                other.id != candidate.id && other.field == .biomarker &&
                (other.value.hasPrefix(candidate.value + " ") || other.value.hasPrefix(candidate.value + ","))
            }
        }
        return results.map { candidate in
            let conflicts = (candidate.field == .condition || candidate.field == .age) && results.filter { $0.field == candidate.field }.count > 1
            return ScanCandidate(field: candidate.field, value: candidate.value, needsReview: candidate.needsReview || conflicts)
        }
    }
    private static func matches(_ pattern: String, _ text: String) -> Bool { text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil }
    private static func capture(_ pattern: String, _ text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let range = Range(match.range(at: 1), in: text) else { return nil }
        return String(text[range])
    }
    static func conflicts(_ candidates: [ScanCandidate], draft: PatientProfileDraft) -> [ScanCandidate] {
        candidates.filter {
            let existing = $0.field == .condition ? draft.condition : $0.field == .age ? draft.ageText : ""
            return !existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && existing.caseInsensitiveCompare($0.value) != .orderedSame
        }
    }
}
