# Heart-Linked TU

## Current revision — reference-based centered heart

`D-centered-heart.svg` is now the production mark. The heart and upper join share an optical center at x=58 in the 100-unit canvas, balancing the letter bodies rather than the extended T crossbar. The transparent heart flows into the gap between the letters. The splash follows the supplied reference with vivid blue radial lighting, a larger white wordmark, and a white capsule Continue button. Compact headers inherit the same vector. Earlier A–C options below are retained for history.


Open `comparison.html` to compare all three transparent vector options at 128, 32, and 24 points on cobalt, light, dark, and monochrome backgrounds.

- **A — Rounded link (selected):** broad supporting U, rounded twin-lobed T crossbar, and gently pointed stem. The heart cue comes from the letter geometry; there is no separate heart symbol. Strongest compact weight and softer health-tech character.
- **B — Flat link:** most literal T; readable but loses the heart cue.
- **C — Soft link:** narrower T stem and lighter U; less substantial at compact sizes.

The production asset is `../../TrialUnity/Assets.xcassets/BrandLogo.imageset/BrandLogo.svg`. It uses the same geometry as A and is rendered as a vector-preserving template. The shared BrandMark supplies adaptive ClinicalAccent in headers and white on the splash. `previous-logo.png` archives the replaced raster outside the app asset catalog.

Splash: #2563EB, 128pt symbol, 28pt semibold wordmark, 18pt gap. Mark fades/scales from 0.94 to 1, then wordmark and Continue fade in. Reduce Motion presents everything immediately.

## Verification (2026-09-30)

- Visually inspected the rendered comparison: splash size, 32pt, 24pt, white on cobalt, adaptive blue on light/dark, and black monochrome. No visible clipping or background container.
- Debug simulator build passed for the project; final incremental asset-catalog build passed without warnings.
- Installed and launched on iPhone 16 Pro (iOS 18.6) and iPhone SE (3rd generation, iOS 18.6).
- Device screenshot verification remains incomplete: CoreSimulator returned `SimDisplayScreenshotWriter.ScreenshotError code=2: Error creating the image` on both devices, including after opening Simulator. Device-specific layout and runtime Reduce Motion still need visual confirmation.
- Reduce Motion is handled in the existing environment branch by setting every splash element visible without animation. Header sizes and alignment are preserved by the shared BrandMark replacement.
