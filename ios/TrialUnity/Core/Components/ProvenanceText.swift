import SwiftUI

/// Monospaced treatment reserved for NCT IDs and data-provenance strings, so
/// they read visually as "identifiers/evidence metadata," not prose.
struct ProvenanceText: View {
    let text: String
    var color: Color = Theme.Color.muted

    var body: some View {
        Text(text)
            .font(.provenance)
            .foregroundStyle(color)
    }
}

#Preview {
    ProvenanceText(text: "NCT00000001 · ClinicalTrials.gov")
        .padding()
}
