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

## Verification on this workspace

Debug simulator and Release device builds passed. All five Swift XCTest tests passed,
including a real URLSession POST to FastAPI returning 10 recommendations with
`source: clinicaltrials.gov`. The generated Debug plist contains only the local-network
ATS exception; the generated Release plist contains no ATS exceptions.
