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
    var originIsProfile: Bool { originLabel.hasPrefix("Profile") }
    /// True once the patient picked "Current location" (until it is denied or switched back).
    var usesCurrentLocation: Bool { requestedLocation }

    private let lookup: ((String) async -> SiteCoordinate?)?
    private let requestAuthorization: (CLLocationManager) -> Void
    private let requestLocation: (CLLocationManager) -> Void

    init(lookup: ((String) async -> SiteCoordinate?)? = nil,
         requestAuthorization: @escaping (CLLocationManager) -> Void = { $0.requestWhenInUseAuthorization() },
         requestLocation: @escaping (CLLocationManager) -> Void = { $0.requestLocation() }) {
        self.lookup = lookup
        self.requestAuthorization = requestAuthorization
        self.requestLocation = requestLocation
        super.init(); manager.delegate = self; manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
    }

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
            if coordinate == nil, let facility = site.facility, !facility.isEmpty { coordinate = await resolvePlace(facility: facility, location: site.location) }
            resolved.append(ResolvedTrialSite(id: index, site: site, coordinate: coordinate))
            sites = resolved
        }
        loaded = true
    }
    private func resolve(_ address: String) async -> SiteCoordinate? {
        if let lookup { return await lookup(address) }
        return await cache.resolve(address) { [self] value in
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
    /// Named-place lookup for sites whose address text didn't geocode. A result is accepted only
    /// when its city appears in the registry's location text, so a look-alike elsewhere is never used.
    private func resolvePlace(facility: String, location: String) async -> SiteCoordinate? {
        guard lookup == nil else { return nil }
        return await cache.resolve("place: \(facility), \(location)") { _ in
            let request = MKLocalSearch.Request()
            request.naturalLanguageQuery = "\(facility), \(location)"
            request.resultTypes = .pointOfInterest
            do {
                let response = try await MKLocalSearch(request: request).start()
                guard let item = response.mapItems.first, let city = item.placemark.locality?.lowercased(),
                      location.lowercased().contains(city), let point = item.placemark.location else { return nil }
                return SiteCoordinate(latitude: point.coordinate.latitude, longitude: point.coordinate.longitude, precise: true)
            } catch { return nil }
        }
    }
    func useCurrentLocation() {
        useCurrentLocation(status: manager.authorizationStatus)
    }
    func useCurrentLocation(status: CLAuthorizationStatus) {
        requestedLocation = true; message = nil
        switch status {
        case .notDetermined: locating = true; requestAuthorization(manager)
        case .authorizedAlways, .authorizedWhenInUse: locating = true; requestLocation(manager)
        default: denied()
        }
    }
    func useProfileLocation(_ label: String?) {
        requestedLocation = false; locating = false; manager.stopUpdatingLocation()
        origin = profileOrigin; originLabel = "Profile: \(label ?? "Not provided")"; message = nil
    }
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        authorizationChanged(manager.authorizationStatus)
    }
    func authorizationChanged(_ status: CLAuthorizationStatus) {
        guard requestedLocation else { return }
        switch status {
        case .authorizedAlways, .authorizedWhenInUse: requestLocation(manager)
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
        message = nil
    }
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        guard requestedLocation else { return }
        locating = false; message = "Couldn't obtain current location. You can still browse trial locations."
    }
    private func denied() {
        locating = false
        requestedLocation = false
        origin = profileOrigin
        originLabel = "Profile location"
        message = "Location access is unavailable. Your profile location and trial locations still work."
    }
    /// Opens Apple Maps with the site preloaded as the destination; Apple Maps supplies the route origin.
    func directions(to resolved: ResolvedTrialSite) {
        guard let point = resolved.coordinate else { return }
        let destination = MKMapItem(placemark: MKPlacemark(coordinate: point.clCoordinate))
        destination.name = resolved.site.facility ?? resolved.site.location
        if !destination.openInMaps(launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving]) {
            message = "Apple Maps couldn't be opened. Please try again."
        }
    }
}

private extension SiteCoordinate {
    var clCoordinate: CLLocationCoordinate2D { .init(latitude: latitude, longitude: longitude) }
}

#if os(iOS)
struct TrialLocationsView: View {
    let trial: Trial
    let profileLocation: String?
    @State private var model = TrialLocationModel()
    @State private var selected: Int?
    @State private var camera: MapCameraPosition = .automatic
    @State private var showsAll = false
    @State private var framed = false
    @State private var visibleSiteCount = 20
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    private var hasCoordinates: Bool { model.sites.contains { $0.coordinate != nil } }
    private var selectedSite: ResolvedTrialSite? { model.sites.first { $0.id == selected } }
    /// Offer the toggle only when "all" would show something the nearby view doesn't.
    private var canToggleScope: Bool {
        SiteViewport.nearby(model.sites, origin: model.origin).count < model.sites.filter { $0.coordinate != nil }.count
    }

    var body: some View {
        VerticalScrollView {
            VStack(alignment: .leading, spacing: Theme.Metrics.sectionSpacing) {
                if hasCoordinates {
                    Map(position: $camera, selection: $selected) {
                        UserAnnotation()
                        if let origin = model.origin, model.originIsProfile {
                            Marker("Your location", systemImage: "person.fill", coordinate: origin.clCoordinate)
                                .tint(Theme.Color.accent)
                        }
                        ForEach(model.sites) { site in
                            if let point = site.coordinate {
                                Marker(site.site.facility ?? site.site.location, coordinate: point.clCoordinate)
                                    .tint(selected == site.id ? Theme.Color.accent : .red)
                                    .tag(site.id)
                            }
                        }
                    }
                    .mapControls { MapUserLocationButton(); MapCompass(); MapScaleView() }
                    .frame(height: 320)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
                    .accessibilityLabel("Trial locations map. Location details are also listed below.")
                    if canToggleScope || showsAll {
                        Button { setScope(all: !showsAll) } label: {
                            actionLabel(showsAll ? "Nearby" : "View all locations", symbol: showsAll ? "location.circle" : "globe")
                        }.buttonStyle(LocationActionStyle())
                    }
                }
                if model.loading { loadingIndicator("Finding trial locations…") }

                CardContainer {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Your starting location").font(.headline)
                            Spacer(minLength: 8)
                            Menu {
                                Picker("Starting location", selection: Binding(
                                    get: { model.usesCurrentLocation },
                                    set: { current in
                                        if current { model.useCurrentLocation() } else { model.useProfileLocation(profileLocation) }
                                    }
                                )) {
                                    Text("Current location").tag(true)
                                    Text("Profile location").tag(false)
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Text(model.usesCurrentLocation ? "Current location" : "Profile location")
                                    Image(systemName: "chevron.up.chevron.down").font(.caption2)
                                }
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Theme.Color.accent)
                                .frame(minHeight: 44)
                            }
                        }
                        Text(model.originLabel).font(.subheadline).foregroundStyle(Theme.Color.muted)
                        if model.locating { loadingIndicator("Finding current location…") }
                        if let message = model.message {
                            Text(message).font(.footnote).foregroundStyle(Theme.Color.muted)
                        }
                    }
                }

                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    Text("Study locations").font(.title3.bold())
                    Text("\(trial.mapSites.count) listed sites").font(.subheadline).foregroundStyle(Theme.Color.muted)
                    DisclosureGroup("About pins and distances") {
                        Text("City pins mark an approximate area. Distances appear only for precise locations and are straight-line estimates, not driving distances.")
                            .font(.caption).foregroundStyle(Theme.Color.muted).padding(.top, 4)
                    }.font(.subheadline)
                }
                if trial.mapSites.isEmpty {
                    Text("No locations reported. Check the registry for updates.")
                        .font(.subheadline).foregroundStyle(Theme.Color.muted)
                }
                if let nearest = model.nearest, !model.loading {
                    Label("Nearest listed site: \(nearest.site.address)", systemImage: "location.circle")
                        .font(.subheadline.weight(.semibold))
                }
                LazyVStack(spacing: 16) {
                    ForEach(model.sites.prefix(visibleSiteCount)) { siteCard($0) }
                }
                if model.sites.count > visibleSiteCount {
                    Button { visibleSiteCount += 20 } label: {
                        actionLabel("Load more locations", symbol: "chevron.down")
                    }.buttonStyle(LocationActionStyle())
                }
                Divider()
                if let url = URL(string: trial.source_url ?? "https://clinicaltrials.gov/study/\(trial.nct_id)") {
                    Link(destination: url) {
                        actionLabel("View registry locations", symbol: "arrow.up.right.square")
                    }.buttonStyle(LocationActionStyle())
                }
            }.padding(Theme.Metrics.screenPadding)
        }
        .background(Theme.Color.paper)
        .navigationTitle("Trial locations").navigationBarTitleDisplayMode(.inline)
        .task { await model.load(trial: trial, profileLocation: profileLocation) }
        .onChange(of: hasCoordinates) { _, has in if has { frameInitial() } }
        .onChange(of: model.loading) { _, loading in if !loading { frameInitial() } }
        .onChange(of: selected) { _, id in focus(on: id) }
        .sheet(isPresented: Binding(get: { selectedSite != nil }, set: { if !$0 { selected = nil } })) {
            if let site = selectedSite { siteSheet(site) }
        }
    }

    // MARK: Camera

    private func region(_ frame: MapFrame) -> MKCoordinateRegion {
        MKCoordinateRegion(center: .init(latitude: frame.latitude, longitude: frame.longitude),
                           span: .init(latitudeDelta: frame.latitudeSpan, longitudeDelta: frame.longitudeSpan))
    }

    /// Opens around the patient / nearby sites -- never the whole world by default.
    private func frameInitial() {
        guard !framed || model.loading == false else { return }
        guard let frame = showsAll ? SiteViewport.allFrame(model.sites) : SiteViewport.nearbyFrame(model.sites, origin: model.origin) else { return }
        framed = true
        camera = .region(region(frame))
    }

    private func setScope(all: Bool) {
        showsAll = all
        selected = nil
        guard let frame = all ? SiteViewport.allFrame(model.sites) : SiteViewport.nearbyFrame(model.sites, origin: model.origin) else { return }
        withAnimation(.easeInOut(duration: 0.6)) { camera = .region(region(frame)) }
    }

    private func focus(on id: Int?) {
        guard let id, let point = model.sites.first(where: { $0.id == id })?.coordinate else { return }
        withAnimation(.easeInOut(duration: 0.5)) {
            camera = .camera(MapCamera(centerCoordinate: point.clCoordinate, distance: 20_000))
        }
    }

    // MARK: Site sheet

    private func siteSheet(_ site: ResolvedTrialSite) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(site.site.facility ?? site.site.location).font(.title3.bold())
            if site.site.facility != nil { Text(site.site.location).font(.subheadline).foregroundStyle(Theme.Color.muted) }
            if let status = site.site.status { TrialStatusText(status: status) }
            if let point = site.coordinate {
                if !point.precise { Text("Approximate area · facility address unverified").font(.caption).foregroundStyle(Theme.Color.attention) }
                if let origin = model.origin, origin.precise, point.precise {
                    Text("About \(origin.meters(to: point) / 1000, specifier: "%.1f") km straight-line").font(.caption).foregroundStyle(Theme.Color.muted)
                }
            }
            Spacer(minLength: 0)
            HStack(spacing: 10) {
                Button { model.directions(to: site) } label: {
                    actionLabel("Directions", symbol: "arrow.triangle.turn.up.right.diamond")
                }.buttonStyle(LocationActionStyle())
                Button { selected = nil } label: {
                    Text("Close").font(.subheadline.weight(.semibold)).frame(maxWidth: .infinity, minHeight: 44)
                        .foregroundStyle(Theme.Color.accent)
                        .background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.control))
                }.buttonStyle(.plain)
            }
        }
        .padding(Theme.Metrics.screenPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .presentationDetents([.height(250)])
        .presentationBackgroundInteraction(.enabled)
        .presentationDragIndicator(.visible)
    }

    private func loadingIndicator(_ title: String) -> some View {
        ProgressView(title)
            .font(.subheadline)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.vertical, 8)
    }

    private func actionLabel(_ title: String, symbol: String) -> some View {
        Label(title, systemImage: symbol)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity, minHeight: 44)
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
                    Button { model.directions(to: site) } label: {
                        actionLabel("Directions", symbol: "arrow.triangle.turn.up.right.diamond")
                    }.buttonStyle(LocationActionStyle())
                } else {
                    Text("Location couldn't be resolved. Check the registry address.").font(.caption).foregroundStyle(Theme.Color.muted)
                }
            }
        }
    }
}

private struct LocationActionStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            .background(Theme.Color.accent, in: RoundedRectangle(cornerRadius: Theme.Radius.control))
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.45)
    }
}
#endif
