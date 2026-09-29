import Foundation

protocol RecommendationsProviding {
    func recommendations(for profile: PatientProfile) async throws -> TrialSearchResponse
}

struct APIClient: RecommendationsProviding {
    let configuration: APIConfiguration
    let session: URLSession

    init(configuration: APIConfiguration = .current, session: URLSession = .shared) {
        self.configuration = configuration
        self.session = session
    }

    func post<Body: Encodable, Response: Decodable>(_ path: String, body: Body) async throws -> Response {
        let url = try configuration.validatedURL().appendingPathComponent(path)
        var request = URLRequest(url: url, timeoutInterval: configuration.timeout)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        do { request.httpBody = try JSONEncoder().encode(body) }
        catch { throw APIError.encoding }
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw APIError.malformedResponse }
            guard (200..<300).contains(http.statusCode) else {
                throw APIError.response(status: http.statusCode, data: data)
            }
            do { return try JSONDecoder().decode(Response.self, from: data) }
            catch { throw APIError.malformedResponse }
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw APIError.map(error)
        }
    }

    func recommendations(for profile: PatientProfile) async throws -> TrialSearchResponse {
        try await post("api/recommendations", body: profile)
    }
}
