import SwiftUI

/// Uses the same finite measurements for sizing and placement. SwiftUI probes
/// layouts with zero, unspecified and infinite proposals; none is a concrete frame.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    private func finite(_ value: CGFloat) -> CGFloat { CGFloat(VisualNumber.dimension(Double(value))) }

    private func arrangement(width: CGFloat?, subviews: Subviews) -> (CGSize, [CGRect]) {
        let gap = finite(spacing)
        let ideal = subviews.map { view -> CGSize in
            let size = view.sizeThatFits(.unspecified)
            return CGSize(width: finite(size.width), height: finite(size.height))
        }
        let natural = finite(ideal.reduce(0) { $0 + $1.width + gap })
        let limit = width.map { $0.isFinite ? finite($0) : natural } ?? natural
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0, used: CGFloat = 0
        var frames: [CGRect] = []
        for (index, view) in subviews.enumerated() {
            let proposedWidth = min(ideal[index].width, limit)
            let measured = view.sizeThatFits(ProposedViewSize(width: proposedWidth, height: nil))
            let size = CGSize(width: min(finite(measured.width), limit), height: finite(measured.height))
            if x > 0, x + size.width > limit { x = 0; y = finite(y + rowHeight + gap); rowHeight = 0 }
            frames.append(CGRect(x: x, y: y, width: size.width, height: size.height))
            used = max(used, finite(x + size.width))
            rowHeight = max(rowHeight, size.height)
            x = finite(x + size.width + gap)
        }
        return (CGSize(width: used, height: finite(y + rowHeight)), frames)
    }
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        arrangement(width: proposal.width, subviews: subviews).0
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let frames = arrangement(width: bounds.width, subviews: subviews).1
        for (view, frame) in zip(subviews, frames) {
            let x = bounds.minX.isFinite ? bounds.minX : 0
            let y = bounds.minY.isFinite ? bounds.minY : 0
            view.place(at: CGPoint(x: x + frame.minX, y: y + frame.minY), proposal: ProposedViewSize(frame.size))
        }
    }
}
