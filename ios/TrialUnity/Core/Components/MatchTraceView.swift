import SwiftUI

/// Reusable structured-signal breakdown (Condition/Age/Sex/Treatment/Location/
/// Phase). Tapping a row expands it to show the patient value, trial value, and
/// source note behind that one signal -- reusing exactly what the backend
/// already returned via MatchTraceBuilder, never recomputed here.
struct MatchTraceView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let signals: [MatchSignal]
    @State private var expandedID: String?

    var body: some View {
        VStack(spacing: 0) {
            ForEach(signals) { signal in
                MatchSignalRow(signal: signal, isExpanded: expandedID == signal.id) {
                    withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.2)) {
                        expandedID = (expandedID == signal.id) ? nil : signal.id
                    }
                }
                if signal.id != signals.last?.id {
                    Divider().overlay(Theme.Color.hairline)
                }
            }
        }
    }
}

private struct MatchSignalRow: View {
    let signal: MatchSignal
    let isExpanded: Bool
    let onToggle: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: onToggle) {
                HStack(spacing: Theme.Spacing.s) {
                    Image(systemName: signal.status.symbolName)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(signal.status.tintColor)
                        .frame(width: 20)
                    Text(signal.label)
                        .font(.body)
                        .foregroundStyle(Theme.Color.ink)
                    Spacer()
                    Text(signal.statusText)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(signal.status.tintColor)
                    Image(systemName: "chevron.down")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(Theme.Color.muted)
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .frame(minHeight: Theme.Metrics.minTapTarget)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("\(signal.label): \(signal.statusText)")
            .accessibilityHint(isExpanded ? "Double tap to collapse detail" : "Double tap for detail")
            .accessibilityAddTraits(.isButton)

            if isExpanded {
                VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                    detailRow(title: "You", value: signal.patientValue)
                    detailRow(title: "Trial", value: signal.trialValue)
                    if let sourceNote = signal.sourceNote {
                        Text(sourceNote)
                            .font(.caption)
                            .foregroundStyle(Theme.Color.muted)
                    }
                }
                .padding(.leading, 28)
                .padding(.bottom, Theme.Spacing.s)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    private func detailRow(title: String, value: String) -> some View {
        HStack(alignment: .top, spacing: Theme.Spacing.s) {
            Text(title.uppercased())
                .font(.caption2.weight(.semibold))
                .foregroundStyle(Theme.Color.muted)
                .frame(width: 44, alignment: .leading)
            Text(value)
                .font(.subheadline)
                .foregroundStyle(Theme.Color.ink)
        }
    }
}

#Preview {
    MatchTraceView(signals: [
        MatchSignal(id: "condition", label: "Condition", status: .match, statusText: "Match", patientValue: "Lung cancer", trialValue: "Lung Cancer", sourceNote: "Trial condition text."),
        MatchSignal(id: "age", label: "Age", status: .unknown, statusText: "Needs review", patientValue: "Not provided", trialValue: "Min 18, max not specified", sourceNote: "Patient age is missing."),
        MatchSignal(id: "sex", label: "Sex", status: .noMatch, statusText: "Conflict found", patientValue: "Male", trialValue: "Female", sourceNote: "Patient sex: MALE; trial sex: FEMALE."),
    ])
    .padding()
}
