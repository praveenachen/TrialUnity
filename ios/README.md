# Native backend integration

Requires Xcode 16 and iOS 17 or later. The Swift package runs focused networking,
DTO, and observable-state tests on macOS without launching a simulator.

## Run locally

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
```

Open `ios/TrialUnity.xcodeproj` in Xcode, select the **TrialUnity** scheme,
**Debug** configuration and an iPhone simulator, then Run. Complete the profile,
review it, and tap **Find matching trials**. The default API URL is
`http://127.0.0.1:8001`. First retrieval can take longer while the backend loads
its embedding model; requests time out after 60 seconds and offer Retry.

To override the Debug URL, add `TRIALUNITY_API_BASE_URL` under
Product → Scheme → Edit Scheme → Run → Arguments → Environment Variables.
For a physical device, bind Uvicorn to `0.0.0.0`, use the Mac's LAN hostname
(e.g. `http://your-mac.local:8001`), join the same network, and allow Local Network
access. Loopback on a physical device refers to the device itself.

The Debug-only plist allows local networking, with no global arbitrary-load
exception. Release has no HTTP exception and requires a real HTTPS URL in the
`TrialUnityAPIBaseURL` Info.plist key (set via
`INFOPLIST_KEY_TrialUnityAPIBaseURL` in Release build settings). There is no
invented production endpoint; absent configuration yields a typed error.

## Verify

```sh
xcodebuild -project ios/TrialUnity.xcodeproj -scheme TrialUnity \
  -configuration Debug -sdk iphonesimulator \
  -derivedDataPath /tmp/TrialUnityBuild CODE_SIGNING_ALLOWED=NO build
swift test --package-path ios --scratch-path /tmp/TrialUnitySwiftTests
# With the backend running, also execute the real URLSession request:
TRIALUNITY_RUN_E2E=1 swift test --package-path ios --scratch-path /tmp/TrialUnitySwiftTests
```

`TrialUnityTests/Fixtures/recommendations.json` is a synthetic contract fixture
serialized by the current backend Pydantic models; it is not a clinical record.
The live test uses the same APIClient and DTO mapping as the app. It prints the
actual response source, including sample-data fallback when used by the backend.

DTOs preserve backend snake_case keys. Missing optional request values are omitted
(Pydantic defaults them to null), undisclosed sex is omitted, treatment choices map
to `intervention_preferences`, notes are retained, and phase preferences are empty.
Travel preference remains local. Relevance, structured eligibility, ESR, and
experimental representation risk are independent types and are never recalculated
by the app. Debug screens show API URL, response source, ESR evidence mode and
component sources, and risk evidence type/model version.

## Results experience and Trial Passport

`MatchingView` shows a real retrieval-funnel summary (`MatchingFunnelSummaryView`,
backed by the backend's `funnel` field -- real counts, never invented) above a
shortlist of `TrialResultRow`s. While a request is in flight, `MatchingFunnelLoadingView`
shows the four conceptual stages with no counts and no fake ticking progress --
the backend computes a search in one request, so there is no honest per-stage
signal to animate before the response arrives.

Tapping a result pushes `TrialPassportView`, a full-screen (not modal) passport
with: trial identity, **Your Match** (a reusable `MatchTraceView` built by
`MatchTraceBuilder` from the six structured signals -- Condition/Age/Sex/Treatment/
Location/Phase -- each row expandable to its patient value, trial value, and
source note), trial overview, eligibility evidence (age/sex bounds, full criteria
text, "needs confirmation" items), **Representation & Access** (`ESRScoreView`:
overall score, an `CoverageBar` for evidence coverage, and a socioeconomic/sex/race
breakdown where a missing component always renders as "—", never 0), a visually
distinct **Predicted Representation Risk** section (`RepresentationRiskView`,
dashed border, "EXPERIMENTAL · PREDICTED · NOT OBSERVED EVIDENCE" label, only
shown when the backend actually returns a prediction -- i.e. never alongside
observed ESR race evidence), and source/provenance links.

`RelevanceTier`, `EligibilitySummary`, `ESREvidenceType`, `ESRDisplay`, and
`MatchTraceBuilder` (all in `Core/Models/`) are pure, SPM-testable presentation
logic -- they only label and group fields the backend already computed; none of
them recalculates a score.

## Verification on this workspace

Xcode.app is not installed in this session's environment (Command Line Tools
only), so a real `xcodebuild`/simulator run could not be performed here -- please
build in Xcode as the first real check. What was verified in this environment:

- `swiftc -typecheck` across all 44 app Swift files (macOS target, with only the
  three genuinely iOS-only calls -- `navigationBarTitleDisplayMode`,
  `keyboardType` -- elided): clean, no errors.
- `project.pbxproj` was regenerated to include five files a prior patch had
  wired in non-standard, ungrouped `SOURCE_ROOT`-relative entries (`APIClient`,
  `APIConfiguration`, `APIError`, `APIModels`, `MatchingModel`) plus every new
  file from this phase; validated with `plutil -lint`, converted to XML with no
  dangling object references, and cross-checked so all 46 Swift files on disk
  match 1:1 with the project's file list and Sources build phase. The scheme's
  `BlueprintIdentifier` was updated to match the regenerated target.
- `swift test` -- **16 of 16 tests pass** (11 new presentation-logic tests plus
  the 5 pre-existing networking/DTO/state tests).
- `TRIALUNITY_RUN_E2E=1 swift test --filter testLocalEndToEnd` against a real
  locally running FastAPI backend: passed, a genuine `URLSession` round trip
  that hit live ClinicalTrials.gov (`source=clinicaltrials.gov`, not the
  sample-data fallback) and decoded 10 real recommendations through the exact
  DTOs the UI consumes, including the new non-optional `funnel` field. A
  follow-up `curl` against the same running backend confirmed a trial with
  observed race enrollment data returns `esr` with a real component score and
  `representation_risk: null`, while a trial without it returns a `risk_level`
  prediction -- the ESR/prediction boundary holds on live data, not just fixtures.
