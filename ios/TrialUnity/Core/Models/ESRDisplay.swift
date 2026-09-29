import Foundation

/// Display ordering and labels for ESR components. Pure presentation mapping --
/// the backend key set (`socioeconomic`, `sex`, `race`) is unchanged.
enum ESRDisplay {
    static let componentOrder = ["socioeconomic", "sex", "race"]

    static func label(for key: String) -> String {
        switch key {
        case "socioeconomic": return "Socioeconomic access"
        case "sex": return "Sex inclusivity"
        case "race": return "Race representation"
        default: return key.capitalized
        }
    }

    static func orderedComponents(_ components: [String: ComponentEvidence]) -> [(key: String, value: ComponentEvidence)] {
        componentOrder.compactMap { key in
            components[key].map { (key, $0) }
        } + components.keys.filter { !componentOrder.contains($0) }.sorted().compactMap { key in
            components[key].map { (key, $0) }
        }
    }
}
