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
                BrandHeader()
                StepProgressView(currentStep: stepNumber, totalSteps: totalSteps)

                VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
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
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(Theme.Metrics.cardPadding)
                .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: 16))

                VStack(alignment: .leading, spacing: Theme.Spacing.l) { content }
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(Theme.Metrics.screenPadding)
        }
        .scrollDismissesKeyboard(.interactively)
        .background {
            LinearGradient(colors: [Theme.Color.surface.opacity(0.55), Theme.Color.paper],
                           startPoint: .top, endPoint: .bottom).ignoresSafeArea()
        }
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: Theme.Spacing.s) {
                PrimaryButton(title: continueTitle, isEnabled: isContinueEnabled, action: onContinue)
                if let onSkip {
                    SecondaryButton(title: "Skip for now", action: onSkip)
                }
            }
            .padding(.horizontal, Theme.Metrics.screenPadding)
            .padding(.top, Theme.Spacing.s)
            .padding(.bottom, Theme.Spacing.m)
            .background(.bar)
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}
