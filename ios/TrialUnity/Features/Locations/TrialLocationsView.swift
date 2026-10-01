import SwiftUI
import MapKit
import CoreLocation
import Observation

@MainActor @Observable final class TrialLocationModel: NSObject, @preconcurrency CLLocationManagerDelegate {
    var sites: [ResolvedTrialSite] = []
    var origin: SiteCoordinate?
    var originLabel = "Profile location"
    var message: String?
    var loading = false
    var locating = false
    private var requestedLocation = false
    private var profileOrigin: SiteCoordinate?
    private let manager = CLLocationManager()
    private let geocoder = CLGeocoder()
    private let cache = GeocodeCache()
    private var loaded = false
    var nearest: ResolvedTrialSite? { SiteDistance.nearest(sites, origin: origin) }

    override init() { super.init(); manager.delegate = self; manager.desiredAccuracy = kCLLocationAccuracyHundredMeters }

    func load(trial: Trial, profileLocation: String?) async {
        guard !loaded, !loading else { return }
        loading = true
        defer { loading = false }
        if let profileLocation, !profileLocation.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            profileOrigin = await resolve(profileLocation)
            if !requestedLocation { origin = profileOrigin; originLabel = "Profile: \(profileLocation)" }
        }
        var resolved: [ResolvedTrialSite] = []
        for (index, site) in trial.mapSites.enumerated() {
            if Task.isCancelled { return }
            var coordinate = site.coordinate
            if coordinate == nil { coordinate = await resolve(site.address) }
            if coordinate == nil, site.address != site.location { coordinate = await resolve(site.location) }
            resolved.append(ResolvedTrialSite(id: index, site: site, coordinate: coordinate))
            sites = resolved
        }
        loaded = true
    }
    private func resolve(_ address: String) async -> SiteCoordinate? {
        await cache.resolve(address) { [self] value in
            // Serial requests, paced to avoid bursts against CLGeocoder's rate limit.
            do {
                try await Task.sleep(for: .seconds(1))
                let places = try await geocoder.geocodeAddressString(value)
                guard places.count == 1, let place = places.first, let location = place.location else { return nil }
                return SiteCoordinate(latitude: location.coordinate.latitude, longitude: location.coordinate.longitude,
                                      precise: place.thoroughfare != nil && place.subThoroughfare != nil)
            } catch { return nil }
        }
    }
    func useCurrentLocation() {
        requestedLocation = true; message = nil
        switch manager.authorizationStatus {
        case .notDetermined: locating = true; manager.requestWhenInUseAuthorization()
        case .authorizedAlways, .authorizedWhenInUse: locating = true; manager.requestLocation()
        default: denied()
        }
    }
    func useProfileLocation(_ label: String?) {
        requestedLocation = false; locating = false; manager.stopUpdatingLocation()
        origin = profileOrigin; originLabel = "Profile: \(label ?? "Not provided")"; message = nil
    }
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        guard requestedLocation else { return }
        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse: manager.requestLocation()
        case .denied, .restricted: denied()
        default: break
        }
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard requestedLocation else { return }
        locating = false
        guard let location = locations.last, location.horizontalAccuracy >= 0, location.horizontalAccuracy <= 1000,
              abs(location.timestamp.timeIntervalSinceNow) < 120 else {
            message = "Current location wasn't accurate enough. Your profile location is still available."; return
        }
        origin = SiteCoordinate(latitude: location.coordinate.latitude, longitude: location.coordinate.longitude, precise: true)
        originLabel = "Current location"
    }
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        locating = false; message = "Couldn't obtain current location. You can still browse trial locations."
    }
    private func denied() {
        locating = false
        requestedLocation = false
        origin = profileOrigin
        originLabel = "Profile location"
        message = "Location access is unavailable. Your profile location and trial locations still work."
    }
    func directions(to resolved: ResolvedTrialSite) {
        guard let point = resolved.coordinate else { return }
        let destination = MKMapItem(placemark: MKPlacemark(coordinate: point.clCoordinate))
        destination.name = resolved.site.address
        if let origin {
            let start = MKMapItem(placemark: MKPlacemark(coordinate: origin.clCoordinate))
            start.name = originLabel
            if !MKMapItem.openMaps(with: [start, destination], launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving]) {
                message = "Apple Maps couldn't be opened. Please try again."
            }
        } else if !destination.openInMaps(launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving]) {
            message = "Apple Maps couldn't be opened. Please try again."
        }
    }
}

private extension SiteCoordinate {
    var clCoordinate: CLLocationCoordinate2D { .init(latitude: latitude, longitude: longitude) }
}

struct TrialLocationsView: View {
    let trial: Trial
    let profileLocation: String?
    @State private var model = TrialLocationModel()
    @State private var selected: Int?
    @State private var camera: MapCameraPosition = .automatic

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                if model.sites.contains(where: { $0.coordinate != nil }) {
                    Map(position: $camera, selection: $selected) {
                        ForEach(model.sites) { site in
                            if let point = site.coordinate {
                                Marker(site.site.facility ?? site.site.location, coordinate: point.clCoordinate).tag(site.id)
                            }
                        }
                    }
                    .frame(height: 280)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
                    .accessibilityLabel("Trial locations map. Location details are also listed below.")
                }
                if model.loading { ProgressView("Finding trial locations…") }
                Text(model.originLabel).font(.subheadline)
                Button("Use my current location", systemImage: "location") { model.useCurrentLocation() }
                    .frame(minHeight: 44).disabled(model.locating)
                if model.locating { ProgressView("Finding current location…") }
                Button("Use profile location") { model.useProfileLocation(profileLocation) }.frame(minHeight: 44)
                if let message = model.message { Text(message).font(.footnote).foregroundStyle(Theme.Color.muted) }
                Text("\(trial.mapSites.count) study locations").font(.headline)
                Text("City-level pins show an area, not an exact facility. Confirm the site address before travel. Distances, when available, are straight-line estimates, not driving distances.")
                    .font(.caption).foregroundStyle(Theme.Color.muted)
                if trial.mapSites.isEmpty { Text("No study locations were reported. Check the registry for updates.") }
                if let nearest = model.nearest, !model.loading {
                    Text("Nearest listed site: \(nearest.site.address)").font(.subheadline.weight(.semibold))
                }
                if let selected, let site = model.sites.first(where: { $0.id == selected }) {
                    siteCard(site)
                }
                ForEach(model.sites.filter { $0.id != selected }) { siteCard($0) }
                if let url = URL(string: trial.source_url ?? "https://clinicaltrials.gov/study/\(trial.nct_id)") {
                    Link("View registry locations", destination: url).frame(minHeight: 44)
                }
            }.padding(Theme.Metrics.screenPadding)
        }
        .background(Theme.Color.paper)
        .navigationTitle("Trial locations").navigationBarTitleDisplayMode(.inline)
        .task { await model.load(trial: trial, profileLocation: profileLocation) }
    }

    private func siteCard(_ site: ResolvedTrialSite) -> some View {
        CardContainer {
            VStack(alignment: .leading, spacing: 8) {
                if let facility = site.site.facility { Text(facility).font(.headline) }
                Text(site.site.location).font(.subheadline)
                if let status = site.site.status { TrialStatusText(status: status) }
                else { Text("Site recruitment not reported; study status: \(trial.status.replacingOccurrences(of: "_", with: " ").capitalized)").font(.caption) }
                if let point = site.coordinate {
                    if !point.precise { Text("Approximate area · facility address unverified").font(.caption).foregroundStyle(Theme.Color.attention) }
                    if let origin = model.origin, origin.precise, point.precise {
                        Text("About \(origin.meters(to: point) / 1000, specifier: "%.1f") km straight-line").font(.caption)
                    }
                    Button("Directions", systemImage: "arrow.triangle.turn.up.right.diamond") { model.directions(to: site) }.frame(minHeight: 44)
                } else {
                    Text("Location couldn't be resolved. Check the registry address.").font(.caption).foregroundStyle(Theme.Color.muted)
                }
            }
        }
    }
}
