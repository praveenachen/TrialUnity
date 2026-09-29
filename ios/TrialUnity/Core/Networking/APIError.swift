import Foundation

enum APIError: Error, LocalizedError, Equatable {
    case configuration
    case timeout
    case unavailable
    case http(status: Int, detail: String?)
    case malformedResponse
    case encoding

    var errorDescription: String? {
        switch self {
        case .configuration: return "The trial service isn't configured."
        case .timeout: return "The search took too long. Please retry."
        case .unavailable: return "Can't reach the trial service. Please retry."
        case .http(let status, _):
            return status == 422 ? "Please check your profile and retry." : "The trial service couldn't complete the search. Please retry."
        case .malformedResponse: return "The trial service returned an unreadable response. Please retry."
        case .encoding: return "Couldn't send your profile. Please retry."
        }
    }

    static func map(_ error: Error) -> APIError {
        if let error = error as? APIError { return error }
        if let error = error as? URLError, error.code == .timedOut { return .timeout }
        return .unavailable
    }
}

private struct BackendError: Decodable {
    let detail: String
}
private struct ValidationError: Decodable {
    struct Issue: Decodable { let msg: String }
    let detail: [Issue]
}

extension APIError {
    static func response(status: Int, data: Data) -> APIError {
        let decoder = JSONDecoder()
        let detail = (try? decoder.decode(BackendError.self, from: data).detail)
            ?? (try? decoder.decode(ValidationError.self, from: data).detail.map(\.msg).joined(separator: "; "))
        return .http(status: status, detail: detail)
    }
}
