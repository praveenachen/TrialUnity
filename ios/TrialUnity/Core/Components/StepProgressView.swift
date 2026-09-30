import SwiftUI

/// A slim segmented progress indicator for the profile wizard, with a text
/// fallback so VoiceOver announces "Step 3 of 7" rather than reading each segment.
struct StepProgressView: View {
    let currentStep: Int
    let totalSteps: Int

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.s) {
            HStack(spacing: 4) {
                ForEach(1...max(totalSteps, 1), id: \.self) { index in
                    Capsule()
                        .fill(index <= currentStep ? Theme.Color.accent : Theme.Color.accent.opacity(0.15))
                        .frame(height: 5)
                }
            }
            Text("Step \(currentStep) of \(totalSteps)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.Color.accent)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Step \(currentStep) of \(totalSteps)")
    }
}

#Preview {
    StepProgressView(currentStep: 3, totalSteps: 7)
        .padding()
}
