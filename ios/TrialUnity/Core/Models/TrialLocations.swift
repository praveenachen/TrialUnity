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

/// A padded bounding box for the map camera.
struct MapFrame: Equatable {
    let latitude: Double
    let longitude: Double
    let latitudeSpan: Double
    let longitudeSpan: Double
}

/// Camera planning for the trial map. Framing only: nothing here is shown to the
/// patient as a distance or as a "nearest" claim.
enum SiteViewport {
    /// Sites within this distance of the anchor count as "nearby".
    static let nearbyRadius: Double = 150_000
    private static let minSpan = 0.08

    /// Sites to frame in the nearby view. The anchor is the patient's location when known,
    /// otherwise the first reliably resolved site (so a global trial still opens locally).
    /// With a patient location and nothing within range, the result is empty -- the camera
    /// stays on the patient rather than jumping to a distant site.
    static func nearby(_ sites: [ResolvedTrialSite], origin: SiteCoordinate?) -> [ResolvedTrialSite] {
        let resolved = sites.filter { $0.coordinate?.isValid == true }
        guard !resolved.isEmpty else { return [] }
        let anchor: SiteCoordinate
        if let origin, origin.isValid { anchor = origin }
        else { anchor = (resolved.first { $0.coordinate?.precise == true } ?? resolved[0]).coordinate! }
        return resolved.filter { anchor.meters(to: $0.coordinate!) <= nearbyRadius }
    }

    /// A padded frame around the points, or nil when there are none. Spans are kept
    /// within MapKit's valid range, so a worldwide trial still produces a usable region.
    static func frame(_ points: [SiteCoordinate], padding: Double = 1.5) -> MapFrame? {
        let valid = points.filter(\.isValid)
        guard let first = valid.first else { return nil }
        var minLat = first.latitude, maxLat = first.latitude, minLon = first.longitude, maxLon = first.longitude
        for point in valid {
            minLat = min(minLat, point.latitude); maxLat = max(maxLat, point.latitude)
            minLon = min(minLon, point.longitude); maxLon = max(maxLon, point.longitude)
        }
        return MapFrame(latitude: (minLat + maxLat) / 2, longitude: (minLon + maxLon) / 2,
                        latitudeSpan: min(170, max(minSpan, (maxLat - minLat) * padding)),
                        longitudeSpan: min(340, max(minSpan, (maxLon - minLon) * padding)))
    }

    /// Frame for the nearby view: nearby sites plus the patient location when known.
    static func nearbyFrame(_ sites: [ResolvedTrialSite], origin: SiteCoordinate?) -> MapFrame? {
        var points = nearby(sites, origin: origin).compactMap(\.coordinate)
        if let origin, origin.isValid { points.append(origin) }
        return frame(points)
    }

    /// Frame for "View all locations": every resolved site.
    static func allFrame(_ sites: [ResolvedTrialSite]) -> MapFrame? {
        frame(sites.compactMap(\.coordinate))
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
