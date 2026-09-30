# UX composition continuation

Baseline: `29bf99f5` (`wip: TrialUnity UX composition pass`). The working tree
was clean before this continuation. The existing layouts were retained.

## Implemented in the baseline

- Cobalt welcome; Home with current search, journey counts and latest save;
  Home / Find / Saved navigation.
- Varied intake controls, with travel included in the location step.
- Matching activity and real completed-search counts; concise result cards.
- Trial Passport with compact At a Glance, What to Confirm, full descriptions
  and eligibility evidence behind disclosures.
- Visual ESR summary; experimental representation details behind disclosure.
- Saved selection, stacked comparison, appointment brief and native sharing.

## Changes in this continuation

- Shared appointment text now uses the same readable relevance label as the
  preview instead of the backend ranking rationale containing internal metrics.
  Existing eligibility evidence, search context, questions and links remain.
- Removed the model-confidence percentage from the experimental-risk view.
  Prediction labels, drivers and limitations remain; model data is unchanged.

## Verification — September 29, 2026

- Debug iOS simulator build: passed using the command in `README.md`.
- Swift package suite: 30 tests, two optional integration tests skipped,
  zero failures. Includes a regression check for the shared brief.
- No backend, retrieval, eligibility, ESR or ML computation changes.

## CoreGraphics warning: not yet reproduced with a backtrace

The previous session's `/tmp/trialunity-numeric-context.log` shows warnings
interleaved with UIKit hardware-keyboard events and inline completion activity.
It does not contain a backtrace, so this is correlation, not proof of an Apple
framework defect or an app-side source.

The existing isolated `/tmp/TrialUnityVisualCheck` harness was reused with
`SCREEN=keyboard` and `CG_NUMERICS_SHOW_BACKTRACE=1`. Its condition field and
FlowLayout match the repository. Programmatic `UIKeyInput.insertText` completed;
the simulator screenshot confirmed the inserted text. No numeric warning was
found in the simulator log for this run. This does not exercise the same path
as physical keyboard events.

Hardware typing via System Events was blocked by macOS:
`osascript is not allowed to send keystrokes. (1002)`.
No warning was suppressed and no speculative layout change was made.

To finish the original reproduction in Xcode:

1. Add `CG_NUMERICS_SHOW_BACKTRACE=1` to the Run scheme environment temporarily.
2. Run TrialUnity on the simulator, open Find, focus Condition, and type with
   the physical keyboard, including accepting/rejecting inline completions.
   Repeat with Notes and appointment context.
3. Capture the first warning's complete stack. If needed, add a symbolic
   breakpoint on `CGPostError` and inspect the arguments and first app frame.
4. Fix an app-side numeric producer only if that stack identifies one; rerun
   the same interaction. A framework-only stack should be investigated with
   a minimal native text-field reproduction before attributing the defect.

The simulator keyboard warning remains an open verification item.
