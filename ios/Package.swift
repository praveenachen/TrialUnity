// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TrialUnityIntegration",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "TrialUnityIntegration", targets: ["TrialUnityIntegration"])],
    targets: [
        .target(name: "TrialUnityIntegration", path: "TrialUnity", exclude: ["Assets.xcassets", "App", "Core/Components", "Core/DesignSystem", "Features/Profile", "Features/Welcome", "Features/Matching/MatchingView.swift", "Resources", "Preview Content"], sources: ["Core/Models", "Core/Networking", "Features/Matching/MatchingModel.swift"]),
        .testTarget(name: "TrialUnityIntegrationTests", dependencies: ["TrialUnityIntegration"], path: "TrialUnityTests", resources: [.copy("Fixtures")])
    ]
)
