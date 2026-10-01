import Foundation

struct TrialSite: Codable {
    let facility: String?
    let location: String
    let status: String?
    let latitude: Double?
    let longitude: Double?
    var coordinate: SiteCoordinate? {
        guard let latitude, let longitude else { return nil }
        let value = SiteCoordinate(latitude: latitude, longitude: longitude, precise: true)
        return value.isValid ? value : nil
    }
    var address: String { [facility, location].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: ", ") }
}

struct SiteCoordinate: Codable, Equatable {
    let latitude: Double
    let longitude: Double
    /// Registry coordinates, device GPS, or an address-level geocode. Never a city centroid.
    let precise: Bool
    var isValid: Bool { latitude.isFinite && longitude.isFinite && abs(latitude) <= 90 && abs(longitude) <= 180 }
    func meters(to other: SiteCoordinate) -> Double {
        let r = Double.pi / 180
        let a = pow(sin((other.latitude - latitude) * r / 2), 2)
            + cos(latitude * r) * cos(other.latitude * r) * pow(sin((other.longitude - longitude) * r / 2), 2)
        return 6_371_000 * 2 * asin(sqrt(min(1, max(0, a))))
    }
}

struct ResolvedTrialSite: Identifiable {
    let id: Int
    let site: TrialSite
    let coordinate: SiteCoordinate?
}

extension Trial {
    var mapSites: [TrialSite] {
        if let sites = trial_sites, !sites.isEmpty { return sites }
        return locations.map { TrialSite(facility: nil, location: $0, status: nil, latitude: nil, longitude: nil) }
    }
}

enum SiteDistance {
    static func nearest(_ sites: [ResolvedTrialSite], origin: SiteCoordinate?) -> ResolvedTrialSite? {
        guard let origin, origin.isValid, origin.precise, !sites.isEmpty,
              sites.allSatisfy({ $0.coordinate?.precise == true && $0.coordinate?.isValid == true }) else { return nil }
        return sites.min { origin.meters(to: $0.coordinate!) < origin.meters(to: $1.coordinate!) }
    }
}

/// Only successful, valid responses are cached; misses can be retried on a later visit.
@MainActor final class GeocodeCache {
    private var values: [String: SiteCoordinate]
    private let file: URL
    init(file: URL = URL.applicationSupportDirectory.appendingPathComponent("TrialUnity/geocodes-v1.json")) {
        self.file = file
        values = (try? JSONDecoder().decode([String: SiteCoordinate].self, from: Data(contentsOf: file))) ?? [:]
    }
    static func key(_ address: String) -> String {
        address.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").lowercased()
    }
    func resolve(_ address: String, lookup: (String) async -> SiteCoordinate?) async -> SiteCoordinate? {
        let key = Self.key(address)
        guard !key.isEmpty else { return nil }
        if let cached = values[key], cached.isValid { return cached }
        guard let result = await lookup(address), result.isValid else { return nil }
        values[key] = result
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(values) { try? data.write(to: file, options: .atomic) }
        return result
    }
}
