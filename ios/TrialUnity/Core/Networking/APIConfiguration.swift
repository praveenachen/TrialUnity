import Foundation

struct APIConfiguration {
    let baseURL: URL?
    var timeout: TimeInterval = 60

    static var current: APIConfiguration {
        #if DEBUG
        let raw = ProcessInfo.processInfo.environment["TRIALUNITY_API_BASE_URL"] ?? "http://127.0.0.1:8001"
        #else
        let raw = Bundle.main.object(forInfoDictionaryKey: "TrialUnityAPIBaseURL") as? String ?? ""
        #endif
        return APIConfiguration(baseURL: URL(string: raw))
    }

    func validatedURL() throws -> URL {
        guard let baseURL, let scheme = baseURL.scheme, baseURL.host != nil,
              ["http", "https"].contains(scheme) else { throw APIError.configuration }
        #if !DEBUG
        guard scheme == "https" else { throw APIError.configuration }
        #endif
        return baseURL
    }
}
