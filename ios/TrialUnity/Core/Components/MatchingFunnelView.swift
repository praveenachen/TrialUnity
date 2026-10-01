import SwiftUI

/// Centered, animated search-in-progress state. Checklist progress comes from
/// MatchingModel (real backend stages, paced client-side only between them);
/// nothing here invents counts or results.
struct MatchingFunnelLoadingView: View {
    var completedSteps = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var spinning = false
    private let steps = [
        "Searching current studies",
        "Checking structured eligibility",
        "Ranking relevant trials",
        "Reviewing representation evidence",
    ]

    var body: some View {
        GeometryReader { geometry in
            VerticalScrollView {
                VStack(spacing: Theme.Spacing.xxl) {
                    VStack(spacing: Theme.Spacing.l) {
                        wheel
                        Text("Finding trials for you")
                            .font(.title2.bold())
                            .multilineTextAlignment(.center)
                        Text("Searching and reviewing studies...")
                            .font(.subheadline)
                            .foregroundStyle(Theme.Color.muted)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .accessibilityElement(children: .combine)

                    VStack(alignment: .leading, spacing: Theme.Spacing.l) {
                        ForEach(Array(steps.enumerated()), id: \.offset) { index, title in
                            row(index: index, title: title)
                        }
                    }
                    .frame(maxWidth: 320, alignment: .leading)
                }
                .padding(Theme.Metrics.screenPadding)
                .frame(maxWidth: .infinity, minHeight: CGFloat(VisualNumber.dimension(Double(geometry.size.height))))
            }
        }
    }

    @ViewBuilder private var wheel: some View {
        if reduceMotion {
            ProgressView().controlSize(.large).tint(Theme.Color.accent)
        } else {
            ZStack {
                Circle().stroke(Theme.Color.accent.opacity(0.15), lineWidth: 6)
                Circle()
                    .trim(from: 0, to: 0.3)
                    .stroke(Theme.Color.accent, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                    .rotationEffect(.degrees(spinning ? 360 : 0))
                    .animation(.linear(duration: 1.1).repeatForever(autoreverses: false), value: spinning)
            }
            .frame(width: 72, height: 72)
            .onAppear { spinning = true }
        }
    }

    private func row(index: Int, title: String) -> some View {
        let done = index < completedSteps
        let current = index == completedSteps
        return HStack(spacing: Theme.Spacing.m) {
            Image(systemName: done ? "checkmark.circle.fill" : (current ? "circle.inset.filled" : "circle"))
                .font(.title3)
                .foregroundStyle(done ? Theme.Color.evidence : (current ? Theme.Color.accent : Theme.Color.muted.opacity(0.6)))
                .contentTransition(.symbolEffect(.replace))
                .accessibilityHidden(true)
            Text(title)
                .font(.subheadline.weight(current ? .semibold : .regular))
                .foregroundStyle(done || current ? Theme.Color.ink : Theme.Color.muted)
        }
        .padding(.horizontal, Theme.Spacing.m)
        .frame(minHeight: 40, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(current ? Theme.Color.accent.opacity(0.08) : .clear, in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
        .animation(.easeInOut(duration: 0.3), value: completedSteps)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title), \(done ? "complete" : (current ? "in progress" : "waiting"))")
    }
}

#Preview("Loading") {
    MatchingFunnelLoadingView(completedSteps: 2)
}
