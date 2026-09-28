"""Reproducible representation-risk training/evaluation experiment.

    python -m representation_risk.run
    python -m representation_risk.run --no-write
    python -m representation_risk.run --out some/other/dir

EXPERIMENTAL, NOT VALIDATED: trained and evaluated entirely on synthetic fixture
data (see representation_risk/README.md). This proves the pipeline's mechanics
work end to end; it is not a claim about real-world predictive performance. No
network access is used or required.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from representation_risk.dataset import build_dataset, train_test_split_dataset
from representation_risk.labels import RISK_LEVELS
from representation_risk.models import evaluate_model, train_models

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _print_summary(dataset_size: int, train_size: int, test_size: int, evaluations: list, importances: dict) -> None:
    print("TrialUnity Representation Risk experiment -- EXPERIMENTAL, NOT VALIDATED")
    print("Trained and evaluated on SYNTHETIC fixture data only (no real historical trials).\n")
    print(f"Dataset: {dataset_size} synthetic historical trials ({train_size} train / {test_size} test)\n")
    print(f"{'model':<20} {'macro_f1':>10} {'balanced_acc':>14}")
    for evaluation in evaluations:
        print(f"{evaluation.name:<20} {evaluation.macro_f1:>10.4f} {evaluation.balanced_accuracy:>14.4f}")
    print("\nNo model is hard-coded as the winner; the backend predictor selects whichever")
    print("scores higher macro F1 on this held-out split.\n")
    for evaluation in evaluations:
        print(f"Confusion matrix -- {evaluation.name} (rows=true, cols=predicted, labels={evaluation.labels}):")
        for row in evaluation.confusion_matrix:
            print("  " + str(row))
    print("\nTop feature importances:")
    for name, importance in importances.items():
        top = sorted(importance.items(), key=lambda item: -item[1])[:5]
        print(f"  {name}: " + ", ".join(f"{feature}={value:.3f}" for feature, value in top))


def _to_markdown(payload: dict) -> str:
    lines = [
        "# Representation Risk Experiment (EXPERIMENTAL, NOT VALIDATED)",
        "",
        "Trained and evaluated on **synthetic fixture data only** -- see `representation_risk/README.md`.",
        "",
        f"Dataset: {payload['dataset_size']} synthetic historical trials "
        f"({payload['train_size']} train / {payload['test_size']} test).",
        "",
        "## Evaluation (no hard-coded winner)",
        "",
        "| model | macro F1 | balanced accuracy |",
        "| --- | --- | --- |",
    ]
    for evaluation in payload["evaluations"]:
        lines.append(f"| {evaluation['name']} | {evaluation['macro_f1']:.4f} | {evaluation['balanced_accuracy']:.4f} |")
    lines.append("")
    for evaluation in payload["evaluations"]:
        lines.append(f"### Confusion matrix -- {evaluation['name']} (labels: {evaluation['labels']})")
        lines.append("")
        for row in evaluation["confusion_matrix"]:
            lines.append(f"- {row}")
        lines.append("")
    lines.append("## Feature importance (top 5 per model)")
    lines.append("")
    for name, importance in payload["feature_importance"].items():
        top = sorted(importance.items(), key=lambda item: -item[1])[:5]
        lines.append(f"- **{name}**: " + ", ".join(f"{feature}={value:.3f}" for feature, value in top))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the representation-risk training/evaluation experiment.")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="Directory to write JSON/Markdown results to.")
    parser.add_argument("--no-write", action="store_true", help="Print only; skip writing result files.")
    args = parser.parse_args()

    dataset = build_dataset()
    train, test = train_test_split_dataset(dataset)
    trained_models = train_models(train)
    evaluations = [evaluate_model(model, test) for model in trained_models]
    importances = {model.name: model.feature_importance for model in trained_models}

    _print_summary(len(dataset), len(train), len(test), evaluations, importances)

    if not args.no_write:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "EXPERIMENTAL_NOT_VALIDATED",
            "data_source": "synthetic_fixture",
            "dataset_size": len(dataset),
            "train_size": len(train),
            "test_size": len(test),
            "risk_levels": list(RISK_LEVELS),
            "evaluations": [
                {
                    "name": evaluation.name,
                    "macro_f1": evaluation.macro_f1,
                    "balanced_accuracy": evaluation.balanced_accuracy,
                    "confusion_matrix": evaluation.confusion_matrix,
                    "labels": evaluation.labels,
                }
                for evaluation in evaluations
            ],
            "feature_importance": importances,
        }
        json_path = out_dir / "latest.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path = out_dir / "latest.md"
        markdown_path.write_text(_to_markdown(payload), encoding="utf-8")
        print(f"\nWrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
