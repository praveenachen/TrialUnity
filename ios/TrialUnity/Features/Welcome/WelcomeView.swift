import SwiftUI

struct WelcomeView: View {
    let onStart: () -> Void

    var body: some View {
        GeometryReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                    Spacer(minLength: Theme.Spacing.xl)

                    VStack(alignment: .leading, spacing: Theme.Spacing.m) {
                        Text("Find trials worth asking about.")
                            .font(.editorialLargeTitle)
                            .foregroundStyle(Theme.Color.ink)
                            .fixedSize(horizontal: false, vertical: true)

                        Text("TrialUnity matches current clinical trials against your profile, then explains why each one is relevant, what's uncertain, and what's known about who trials like it have actually enrolled.")
                            .font(.body)
                            .foregroundStyle(Theme.Color.muted)
                    }

                    VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                        Label("Navigation support, not medical advice.", systemImage: "info.circle")
                            .font(.footnote)
                            .foregroundStyle(Theme.Color.muted)
                        Text("SOURCE: CLINICALTRIALS.GOV")
                            .font(.provenance)
                            .foregroundStyle(Theme.Color.muted)
                    }

                    Spacer(minLength: Theme.Spacing.l)

                    PrimaryButton(title: "Find my matches", action: onStart)
                        .accessibilityHint("Starts your guided profile")
                }
                .padding(Theme.Spacing.l)
                .frame(minHeight: proxy.size.height, alignment: .top)
            }
        }
        .background(Theme.Color.paper)
    }
}

#Preview {
    WelcomeView(onStart: {})
}
