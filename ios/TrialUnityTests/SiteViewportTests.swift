import XCTest
@testable import TrialUnityIntegration

final class SiteViewportTests: XCTestCase {
    private func site(_ id: Int, _ lat: Double?, _ lon: Double?, precise: Bool = true) -> ResolvedTrialSite {
        let trialSite = TrialSite(facility: "Site \(id)", location: "City \(id)", status: nil, latitude: lat, longitude: lon)
        let coordinate = lat.flatMap { la in lon.map { SiteCoordinate(latitude: la, longitude: $0, precise: precise) } }
        return ResolvedTrialSite(id: id, site: trialSite, coordinate: coordinate)
    }
    private let toronto = SiteCoordinate(latitude: 43.65, longitude: -79.38, precise: false)

    func testTorontoProfileFramesNearbySitesNotTheWorld() {
        let sites = [site(0, 43.66, -79.39), site(1, 43.70, -79.40), site(2, 35.68, 139.69), site(3, -33.87, 151.21)]
        XCTAssertEqual(SiteViewport.nearby(sites, origin: toronto).map(\.id), [0, 1])
        let frame = try! XCTUnwrap(SiteViewport.nearbyFrame(sites, origin: toronto))
        XCTAssertEqual(frame.latitude, 43.675, accuracy: 0.05)
        XCTAssertLessThan(frame.longitudeSpan, 1)        // local, not world-wide
        let all = try! XCTUnwrap(SiteViewport.allFrame(sites))
        XCTAssertGreaterThan(all.longitudeSpan, 100)     // "View all locations" fits everything
        XCTAssertLessThanOrEqual(all.longitudeSpan, 340)
    }

    func testNoPatientLocationOpensAroundFirstPreciseSite() {
        let sites = [site(0, 35.68, 139.69, precise: false), site(1, 43.66, -79.39), site(2, 43.70, -79.40), site(3, -33.87, 151.21)]
        XCTAssertEqual(SiteViewport.nearby(sites, origin: nil).map(\.id), [1, 2])
    }

    func testPatientFarFromEverySiteStaysOnPatient() {
        let sites = [site(0, 35.68, 139.69), site(1, -33.87, 151.21)]
        XCTAssertTrue(SiteViewport.nearby(sites, origin: toronto).isEmpty)
        let frame = try! XCTUnwrap(SiteViewport.nearbyFrame(sites, origin: toronto))
        XCTAssertEqual(frame.latitude, toronto.latitude, accuracy: 0.001)
    }

    func testUnresolvedAndInvalidSitesAreIgnored() {
        let sites = [site(0, nil, nil), site(1, 91, 0)]
        XCTAssertTrue(SiteViewport.nearby(sites, origin: nil).isEmpty)
        XCTAssertNil(SiteViewport.nearbyFrame(sites, origin: nil))
        XCTAssertNil(SiteViewport.allFrame(sites))
    }

    func testSingleAndDuplicateSitesGetAUsableMinimumSpan() {
        let one = try! XCTUnwrap(SiteViewport.allFrame([site(0, 43.66, -79.39)]))
        XCTAssertGreaterThan(one.latitudeSpan, 0.05)
        let dup = try! XCTUnwrap(SiteViewport.allFrame([site(0, 43.66, -79.39), site(1, 43.66, -79.39)]))
        XCTAssertEqual(dup, one)
    }

    func testListGroupsNearbyFirstThenByDistanceThenUnplaced() {
        let sites = [site(0, 35.68, 139.69), site(1, 43.70, -79.40), site(2, nil, nil), site(3, 43.66, -79.39), site(4, 40.71, -74.00)]
        let groups = SiteOrdering.groups(sites, origin: toronto)
        XCTAssertEqual(groups.nearby.map(\.id), [3, 1])                // closest first
        XCTAssertEqual(groups.others.map(\.id), [4, 0, 2])             // then by distance, unplaced last
    }

    func testListKeepsRegistryOrderWithoutAStartingPoint() {
        let sites = [site(0, 35.68, 139.69), site(1, 43.70, -79.40), site(2, nil, nil)]
        let groups = SiteOrdering.groups(sites, origin: nil)
        XCTAssertTrue(groups.nearby.isEmpty)
        XCTAssertEqual(groups.others.map(\.id), [0, 1, 2])
    }

    func testLookupOrderPrioritizesSitesMatchingTheProfileLocation() {
        func plain(_ location: String) -> TrialSite { TrialSite(facility: nil, location: location, status: nil, latitude: nil, longitude: nil) }
        let sites = [plain("Tokyo, Japan"), plain("Toronto, Ontario, Canada"), plain("Paris, France"), plain("Ottawa, Ontario, Canada")]
        XCTAssertEqual(SiteOrdering.lookupOrder(sites, pending: [0, 1, 2, 3], profileLocation: "Toronto, Ontario"), [1, 3, 0, 2])
        XCTAssertEqual(SiteOrdering.lookupOrder(sites, pending: [0, 1, 2, 3], profileLocation: nil), [0, 1, 2, 3])
    }
}
