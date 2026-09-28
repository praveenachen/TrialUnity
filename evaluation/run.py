"""Reproducible evaluation runner.

    python -m evaluation.run
    python -m evaluation.run --no-write
    python -m evaluation.run --out some/other/dir

Runs the lexical-vs-dense-vs-hybrid ablation over the local benchmark fixture,
plus a handful of ESR structural sanity checks, and prints a concise summary.
Everything runs against local fixtures and the same locally-cached
SentenceTransformer the app uses -- no network access is required or used.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.ablation import run_ablation
from evaluation.esr_checks import run_esr_checks

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _print_summary(result: dict, esr_checks: dict) -> None:
    print(f"TrialUnity retrieval evaluation -- {result['num_cases']} benchmark cases, K={result['ks']}\n")
    metric_names = list(result["aggregate"]["hybrid"].keys())
    print(f"{'system':<8} " + " ".join(f"{name:>10}" for name in metric_names))
    for system in ("lexical", "dense", "hybrid"):
        row = result["aggregate"][system]
        print(f"{system:<8} " + " ".join(f"{row[name]:>10.4f}" for name in metric_names))

    print("\nESR sanity checks (structural only -- not scored against relevance):")
    print(f"  deterministic: {esr_checks['deterministic']}")
    print(f"  missing evidence lowers coverage: {esr_checks['missing_evidence']['coverage_decreased']}")
    print(f"  missing evidence does not force score to zero: {esr_checks['missing_evidence']['score_not_forced_to_zero']}")
    print(f"  does not change clinical ordering: {esr_checks['does_not_change_clinical_ordering']}")


def _to_markdown(result: dict, esr_checks: dict) -> str:
    metric_names = list(result["aggregate"]["hybrid"].keys())
    lines = [
        "# TrialUnity Retrieval Evaluation",
        "",
        f"{result['num_cases']} benchmark cases, K={result['ks']}.",
        "",
        "## Aggregate metrics (mean over all cases)",
        "",
        "| system | " + " | ".join(metric_names) + " |",
        "| --- | " + " | ".join("---" for _ in metric_names) + " |",
    ]
    for system in ("lexical", "dense", "hybrid"):
        row = result["aggregate"][system]
        lines.append(f"| {system} | " + " | ".join(f"{row[name]:.4f}" for name in metric_names) + " |")

    lines += [
        "",
        "## Per-category results",
        "",
        "| case | category | lexical nDCG@5 | dense nDCG@5 | hybrid nDCG@5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['category']} | "
            f"{case['metrics']['lexical']['ndcg@5']:.4f} | "
            f"{case['metrics']['dense']['ndcg@5']:.4f} | "
            f"{case['metrics']['hybrid']['ndcg@5']:.4f} |"
        )

    lines += [
        "",
        "## ESR sanity checks (separate from IR quality -- see evaluation/README.md)",
        "",
        f"- Deterministic: {esr_checks['deterministic']}",
        f"- Missing evidence lowers coverage, not score: "
        f"{esr_checks['missing_evidence']['coverage_decreased']} / "
        f"{esr_checks['missing_evidence']['score_not_forced_to_zero']}",
        f"- Does not change clinical retrieval ordering: {esr_checks['does_not_change_clinical_ordering']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrialUnity's retrieval evaluation benchmark.")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="Directory to write JSON/Markdown results to.")
    parser.add_argument("--no-write", action="store_true", help="Print only; skip writing result files.")
    args = parser.parse_args()

    result = run_ablation()
    esr_checks = run_esr_checks()
    _print_summary(result, esr_checks)

    if not args.no_write:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ablation": result,
            "esr_checks": esr_checks,
        }
        json_path = out_dir / "latest.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path = out_dir / "latest.md"
        markdown_path.write_text(_to_markdown(result, esr_checks), encoding="utf-8")
        print(f"\nWrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
