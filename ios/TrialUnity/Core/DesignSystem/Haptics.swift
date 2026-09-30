import UIKit

/// Two purposeful haptic cues -- not decoration. Reduce Motion governs visual
/// animation, not haptics, so these fire regardless of that setting; iOS itself
/// already respects the system-wide "System Haptics" toggle for these generators.
enum Haptics {
    /// Matching finished and results are ready.
    static func matchingCompleted() {
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }

    /// A trial was saved or unsaved.
    static func saveToggled() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    /// A trial was selected or deselected for comparison.
    static func selectionChanged() {
        UISelectionFeedbackGenerator().selectionChanged()
    }
}
