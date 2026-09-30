import Foundation

/// Safe formatting for backend score values headed into CGFloat/frame/progress
/// calculations. `nil` and non-finite (NaN/±infinity) are both treated as
/// "unavailable" -- never coerced into a number, and never force-converted in a
/// way that could crash or produce a non-finite value for SwiftUI/CoreGraphics.
enum ScoreFormat {
    /// A 0-100 score rounded to the nearest whole number, or "—" if unavailable.
    static func rounded(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        return "\(Int(min(max(value, 0), 100).rounded()))"
    }

    /// Clamps to `range` and guarantees a finite result -- for anything that
    /// feeds a frame width, progress fraction, or similar CoreGraphics-bound value.
    static func clamped(_ value: Double, to range: ClosedRange<Double> = 0...1) -> Double {
        guard value.isFinite else { return range.lowerBound }
        return min(max(value, range.lowerBound), range.upperBound)
    }
}
