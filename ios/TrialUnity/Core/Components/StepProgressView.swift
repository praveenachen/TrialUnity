import SwiftUI

/// A slim segmented progress indicator for the profile wizard, with a text
/// fallback so VoiceOver announces "Step 3 of 7" rather than reading each segment.
struct StepProgressView: View {
    let currentStep: Int
    let totalSteps: Int

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack(spacing: 4) {
                ForEach(1...max(totalSteps, 1), id: \.self) { index in
                    Capsule()
                        .fill(index <= currentStep ? Theme.Color.ink : Theme.Color.hairline)
                        .frame(height: 3)
                }
            }
            Text("Step \(currentStep) of \(totalSteps)")
                .font(.caption)
                .foregroundStyle(Theme.Color.muted)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Step \(currentStep) of \(totalSteps)")
    }
}

#Preview {
    StepProgressView(currentStep: 3, totalSteps: 7)
        .padding()
}
