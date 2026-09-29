import SwiftUI

/// Shared layout for every profile-wizard step: progress indicator, editorial
/// title/subtitle, scrollable content, and a keyboard-safe pinned footer with
/// Continue (and, for optional steps, Skip). Back navigation is the system back
/// button provided by NavigationStack -- deliberately not reimplemented here.
struct WizardScaffold<Content: View>: View {
    let stepNumber: Int
    let totalSteps: Int
    let title: String
    var subtitle: String?
    var isOptional: Bool = false
    var continueTitle: String = "Continue"
    var isContinueEnabled: Bool = true
    var onSkip: (() -> Void)?
    let onContinue: () -> Void
    @ViewBuilder var content: Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.l) {
                StepProgressView(currentStep: stepNumber, totalSteps: totalSteps)

                VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                    HStack(alignment: .firstTextBaseline, spacing: Theme.Spacing.s) {
                        Text(title)
                            .font(.editorialTitle)
                            .foregroundStyle(Theme.Color.ink)
                        if isOptional {
                            Text("Optional")
                                .font(.sectionLabel)
                                .foregroundStyle(Theme.Color.muted)
                        }
                    }
                    if let subtitle {
                        Text(subtitle)
                            .font(.subheadline)
                            .foregroundStyle(Theme.Color.muted)
                    }
                }

                content
            }
            .padding(Theme.Spacing.l)
        }
        .scrollDismissesKeyboard(.interactively)
        .background(Theme.Color.paper)
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: Theme.Spacing.s) {
                PrimaryButton(title: continueTitle, isEnabled: isContinueEnabled, action: onContinue)
                if let onSkip {
                    SecondaryButton(title: "Skip for now", action: onSkip)
                }
            }
            .padding(.horizontal, Theme.Spacing.l)
            .padding(.top, Theme.Spacing.s)
            .padding(.bottom, Theme.Spacing.m)
            .background(.bar)
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}
