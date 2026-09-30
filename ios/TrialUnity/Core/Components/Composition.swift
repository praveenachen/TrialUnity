import SwiftUI

struct DecisionAnchor: View {
    let symbol: String
    let caption: String
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: symbol).font(.largeTitle).foregroundStyle(Theme.Color.accent)
                .frame(width: 64, height: 64).background(Theme.Color.surface, in: RoundedRectangle(cornerRadius: 20))
            Text(caption).font(.subheadline).foregroundStyle(Theme.Color.muted)
        }.accessibilityElement(children: .combine)
    }
}

struct GlanceGrid: View {
    let items: [(String, String)]
    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), alignment: .leading)], alignment: .leading, spacing: 16) {
            ForEach(items.indices, id: \.self) { index in
                VStack(alignment: .leading, spacing: 4) {
                    Text(items[index].0).font(.caption).foregroundStyle(Theme.Color.muted)
                    Text(items[index].1).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.Color.ink)
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}
