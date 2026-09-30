import Foundation

protocol RecommendationsProviding {
    func recommendations(for profile: PatientProfile) async throws -> TrialSearchResponse
    func recommendations(for profile: PatientProfile, progress: @escaping @MainActor (Int) -> Void) async throws -> TrialSearchResponse
}

extension RecommendationsProviding {
    func recommendations(for profile: PatientProfile, progress: @escaping @MainActor (Int) -> Void) async throws -> TrialSearchResponse {
        let response = try await recommendations(for: profile)
        await progress(3)
        return response
    }
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

    func recommendations(for profile: PatientProfile, progress: @escaping @MainActor (Int) -> Void) async throws -> TrialSearchResponse {
        struct Event: Decodable {
            let completed: Int?
            let response: TrialSearchResponse?
            let error: String?
        }
        let url = try configuration.validatedURL().appendingPathComponent("api/recommendations/stream")
        var request = URLRequest(url: url, timeoutInterval: configuration.timeout)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/x-ndjson", forHTTPHeaderField: "Accept")
        do { request.httpBody = try JSONEncoder().encode(profile) }
        catch { throw APIError.encoding }
        do {
            let (bytes, response) = try await session.bytes(for: request)
            guard let http = response as? HTTPURLResponse else { throw APIError.malformedResponse }
            // Older servers still work, without inventing intermediate progress.
            if http.statusCode == 404 || http.statusCode == 405 {
                let response = try await recommendations(for: profile)
                await progress(3)
                return response
            }
            guard (200..<300).contains(http.statusCode) else {
                throw APIError.http(status: http.statusCode, detail: nil)
            }
            for try await line in bytes.lines {
                try Task.checkCancellation()
                if line.isEmpty { continue }
                guard let event = try? JSONDecoder().decode(Event.self, from: Data(line.utf8)) else {
                    throw APIError.malformedResponse
                }
                if event.error != nil { throw APIError.unavailable }
                if let completed = event.completed { await progress(completed) }
                if let response = event.response { return response }
            }
            throw APIError.malformedResponse
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw APIError.map(error)
        }
    }
}
