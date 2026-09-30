import SwiftUI

/// A saved-trial summary row -- the same bordered-card language as
/// TrialResultRow (relevance tier, ESR read) plus when it was saved, so Saved
/// Trials reads like a continuation of Results rather than a different visual
/// language.
struct SavedTrialRow: View {
    let record: SavedTrial
    var showsDisclosure = false

    var body: some View {
        CardContainer(padding: Theme.Spacing.m) {
            VStack(alignment: .leading, spacing: Theme.Spacing.s) {
                HStack(alignment: .top) {
                    FlowLayout(spacing: Theme.Spacing.xs) {
                        ProvenanceText(text: record.id)
                        if let date = record.savedAt {
                            Text("·").foregroundStyle(Theme.Color.muted)
                            Text("Saved \(date.formatted(date: .abbreviated, time: .omitted))")
                                .font(.caption)
                                .foregroundStyle(Theme.Color.muted)
                        }
                    }
                    Spacer(minLength: Theme.Spacing.s)
                    Image(systemName: "bookmark.fill")
                        .font(.caption)
                        .foregroundStyle(Theme.Color.accent)
                        .accessibilityHidden(true)
                }

                HStack(spacing: Theme.Spacing.s) {
                    Text(record.displayTitle)
                        .font(.editorialHeadline)
                        .foregroundStyle(Theme.Color.ink)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if showsDisclosure {
                        CardNavigationArrow()
                    }
                }

                if let result = record.recommendation {
                    TrialStatusText(status: result.trial.status)
                    if let location = PatientPresentation.location(result.trial, near: record.profile?.location) {
                        Label(location, systemImage: "mappin.and.ellipse")
                            .font(.caption)
                            .foregroundStyle(Theme.Color.muted)
                            .lineLimit(1)
                    }
                    FlowLayout(spacing: Theme.Spacing.s) {
                        let tier = RelevanceTier(score: result.score)
                        StatusPill(text: tier.label, symbolName: tier.symbolName, tint: tier.tintColor)
                        StatusPill(
                            text: "Representation & access · \(ScoreFormat.rounded(result.esr?.score))",
                            symbolName: "shield.checkerboard",
                            tint: Theme.Color.muted
                        )
                    }
                } else {
                    Label("Details unavailable -- find this trial again and save it to refresh.", systemImage: "exclamationmark.circle")
                        .font(.caption)
                        .foregroundStyle(Theme.Color.attention)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    List {
        SavedTrialRow(record: SavedTrial(
            id: "NCT00000001", title: "Phase 2 Study of Trastuzumab Deruxtecan in Metastatic Breast Cancer",
            savedAt: .now, source: "clinicaltrials.gov", profile: nil, recommendation: nil
        ))
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
    }
}
