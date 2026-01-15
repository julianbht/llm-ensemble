#!/usr/bin/env python3
"""Combine two confusion matrix analysis tables for side-by-side comparison.

Usage:
    python scripts/figures/combine_confusion_tables.py
"""

import sys
from pathlib import Path

# Add src to path for imports before other imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

import typer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

# Import evaluate startup to ensure all ORMs are registered with SQLAlchemy
from llm_ensemble.evaluate.startup import dependency_configurator  # noqa: F401
from llm_ensemble.evaluate.adapters.driven.io_factory import IOAdapterFactory


# ============================================================================
# CONFIGURATION - Edit these constants to change the comparison
# ============================================================================

# First run (e.g., ensemble)
RUN_1_NAME = "5-ensemble-size-analysis-majority_vote_average-params_small_to_big"
RUN_1_TYPE = "aggregate"
RUN_1_LABEL = "Ensemble"

# Second run (e.g., reference model)
RUN_2_NAME = "reference-ensemble-gpt-5-1-all-samples-start"
RUN_2_TYPE = "infer"
RUN_2_LABEL = "GPT-5.1"

# Output filename prefix
OUTPUT_PREFIX = "confusion_matrix_comparison"

# ============================================================================


app = typer.Typer()


def load_evaluation_data(run_name: str, run_type: str = "aggregate"):
    """Load ground truth and predictions from a run."""
    io_name = "db_aggregate_to_json" if run_type == "aggregate" else "db_infer_to_json"
    reader = IOAdapterFactory.create_reader(io_name)
    eval_data = reader.read(run_name)

    ground_truth = [int(gt) for gt in eval_data.ground_truth]
    predictions = [
        int(pred) if pred is not None else -1 for pred in eval_data.predictions
    ]

    return ground_truth, predictions


def compute_metrics(ground_truth: list, predictions: list) -> dict:
    """Compute all metrics for a run."""
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    gt_valid = [ground_truth[i] for i in valid_indices]
    pred_valid = [predictions[i] for i in valid_indices]

    labels = [0, 1, 2, 3]
    class_names = ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]

    accuracy = accuracy_score(gt_valid, pred_valid)
    recalls = recall_score(gt_valid, pred_valid, labels=labels, average=None, zero_division=0)
    precisions = precision_score(gt_valid, pred_valid, labels=labels, average=None, zero_division=0)

    over_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p > g)
    under_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p < g)
    soft_disagreement = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) == 1)
    hard_disagreement = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) >= 2)

    n = len(gt_valid)

    return {
        "n": n,
        "accuracy": accuracy,
        "recalls": dict(zip(class_names, recalls)),
        "precisions": dict(zip(class_names, precisions)),
        "over_predict": over_predict / n,
        "under_predict": under_predict / n,
        "soft_disagreement": soft_disagreement / n,
        "hard_disagreement": hard_disagreement / n,
    }


def save_comparison_table_txt(metrics1: dict, metrics2: dict, label1: str, label2: str, output_path: Path):
    """Save comparison as a text table."""
    lines = []
    lines.append(f"Confusion Matrix Comparison: {label1} vs {label2}")
    lines.append("")
    lines.append(f"{'Metric':<35} {label1:>15} {label2:>15}")
    lines.append("-" * 67)

    lines.append(f"{'Accuracy':<35} {metrics1['accuracy']:>14.1%} {metrics2['accuracy']:>14.1%}")
    lines.append("")

    for cls in ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]:
        lines.append(f"{'Recall (' + cls + ')':<35} {metrics1['recalls'][cls]:>14.1%} {metrics2['recalls'][cls]:>14.1%}")

    lines.append("")

    for cls in ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]:
        lines.append(f"{'Precision (' + cls + ')':<35} {metrics1['precisions'][cls]:>14.1%} {metrics2['precisions'][cls]:>14.1%}")

    lines.append("")
    lines.append(f"{'Over-predicting (LLM > Human)':<35} {metrics1['over_predict']:>14.1%} {metrics2['over_predict']:>14.1%}")
    lines.append(f"{'Under-predicting (LLM < Human)':<35} {metrics1['under_predict']:>14.1%} {metrics2['under_predict']:>14.1%}")

    lines.append("")
    lines.append(f"{'Soft Disagreement':<35} {metrics1['soft_disagreement']:>14.1%} {metrics2['soft_disagreement']:>14.1%}")
    lines.append(f"{'Hard Disagreement':<35} {metrics1['hard_disagreement']:>14.1%} {metrics2['hard_disagreement']:>14.1%}")

    output_path.write_text("\n".join(lines))
    typer.echo(f"Text table saved to: {output_path}")


def save_comparison_table_latex(metrics1: dict, metrics2: dict, label1: str, label2: str, output_path: Path):
    """Save comparison as a LaTeX table."""
    lines = []

    lines.append("% Confusion Matrix Comparison Table")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Confusion Matrix Comparison: {label1} vs {label2}}}")
    lines.append("\\label{tab:confusion-comparison}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append(f"Metric & {label1} & {label2} \\\\")
    lines.append("\\midrule")

    lines.append(f"Accuracy & {metrics1['accuracy']:.1%} & {metrics2['accuracy']:.1%} \\\\".replace("%", "\\%"))
    lines.append("\\midrule")

    for cls in ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]:
        lines.append(f"Recall ({cls}) & {metrics1['recalls'][cls]:.1%} & {metrics2['recalls'][cls]:.1%} \\\\".replace("%", "\\%"))

    lines.append("\\midrule")

    for cls in ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]:
        lines.append(f"Precision ({cls}) & {metrics1['precisions'][cls]:.1%} & {metrics2['precisions'][cls]:.1%} \\\\".replace("%", "\\%"))

    lines.append("\\midrule")
    lines.append(f"Over-predicting (LLM > Human) & {metrics1['over_predict']:.1%} & {metrics2['over_predict']:.1%} \\\\".replace("%", "\\%"))
    lines.append(f"Under-predicting (LLM < Human) & {metrics1['under_predict']:.1%} & {metrics2['under_predict']:.1%} \\\\".replace("%", "\\%"))

    lines.append("\\midrule")
    lines.append(f"Soft Disagreement & {metrics1['soft_disagreement']:.1%} & {metrics2['soft_disagreement']:.1%} \\\\".replace("%", "\\%"))
    lines.append(f"Hard Disagreement & {metrics1['hard_disagreement']:.1%} & {metrics2['hard_disagreement']:.1%} \\\\".replace("%", "\\%"))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path.write_text("\n".join(lines))
    typer.echo(f"LaTeX table saved to: {output_path}")


@app.command()
def main(
    run1_name: str = typer.Option(RUN_1_NAME, "--run1-name", help="First run name"),
    run1_type: str = typer.Option(RUN_1_TYPE, "--run1-type", help="First run type: aggregate or infer"),
    run1_label: str = typer.Option(RUN_1_LABEL, "--run1-label", help="Label for first run"),
    run2_name: str = typer.Option(RUN_2_NAME, "--run2-name", help="Second run name"),
    run2_type: str = typer.Option(RUN_2_TYPE, "--run2-type", help="Second run type: aggregate or infer"),
    run2_label: str = typer.Option(RUN_2_LABEL, "--run2-label", help="Label for second run"),
):
    """Combine two confusion matrix tables for comparison."""
    project_root = Path(__file__).parent.parent.parent
    tables_dir = project_root / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading {run1_label}: {run1_name}")
    gt1, pred1 = load_evaluation_data(run1_name, run1_type)
    metrics1 = compute_metrics(gt1, pred1)
    typer.echo(f"  Loaded {metrics1['n']:,} samples")

    typer.echo(f"Loading {run2_label}: {run2_name}")
    gt2, pred2 = load_evaluation_data(run2_name, run2_type)
    metrics2 = compute_metrics(gt2, pred2)
    typer.echo(f"  Loaded {metrics2['n']:,} samples")

    # Save outputs
    txt_path = tables_dir / f"{OUTPUT_PREFIX}.txt"
    latex_path = tables_dir / f"{OUTPUT_PREFIX}.tex"

    save_comparison_table_txt(metrics1, metrics2, run1_label, run2_label, txt_path)
    save_comparison_table_latex(metrics1, metrics2, run1_label, run2_label, latex_path)

    typer.echo("\nDone!")


if __name__ == "__main__":
    app()
