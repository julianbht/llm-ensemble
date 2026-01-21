#!/usr/bin/env python3
"""Plot confusion matrix comparing LLM predictions vs human ground truth.

This script reads aggregate run data from the database and creates a confusion
matrix showing how well the LLM ensemble agrees with human annotators.

Answers key questions:
- Are we systematically over/under-predicting relevance?
- Do we confuse adjacent levels (2 vs 3) or distant ones (0 vs 3)?

Usage:
    # Run with defaults from constants below
    python scripts/figures/plot_confusion_matrix.py

    # Override run name
    python scripts/figures/plot_confusion_matrix.py --run-name my-run
"""

import sys
from pathlib import Path

# Add src to path for imports before other imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import typer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

from thesis_colors import (
    BHT_COLORS,
    FONTSIZE,
    FONTSIZE_SMALL,
    FIGURE_WIDTH,
    apply_thesis_style,
)
from copy_to_overleaf import copy_figure_to_overleaf

apply_thesis_style()

# Create custom BHT blue colormap (white to BHT blue)
BHT_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "bht_blue", ["white", BHT_COLORS["blue"]]
)

# Import evaluate startup to ensure all ORMs are registered with SQLAlchemy
from llm_ensemble.evaluate.startup import dependency_configurator  # noqa: F401
from llm_ensemble.evaluate.adapters.driven.io_factory import IOAdapterFactory
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================

# Run name to analyze
# RUN_NAME = "5-ensemble-size-analysis-majority_vote_average-params_small_to_big"
RUN_NAME = "reference-ensemble-gpt-5-1-all-samples-start"

# Run type: "aggregate" or "infer"
# RUN_TYPE = "aggregate"
RUN_TYPE = "infer"

# Labels for the relevance scores (0, 1, 2, 3)
RELEVANCE_LABELS = [
    "0\nIrrelevant",
    "1\nRelevant",
    "2\nHighly\nRelevant",
    "3\nPerfectly\nRelevant",
]

# Short labels for compact display
RELEVANCE_LABELS_SHORT = ["0", "1", "2", "3"]

# Output filename (None = auto-generate)
OUTPUT_FILENAME = None  # e.g., "confusion_matrix.svg" or None

# Custom plot title (None = auto-generate)
# PLOT_TITLE = "Ensemble vs Human Labels"
PLOT_TITLE = "GPT 5.1 vs Human Labels"

# Normalize confusion matrix: None, "true" (row), "pred" (column), "all"
NORMALIZE = None  # None for counts, "true" for recall per class

# Color map for the heatmap (uses custom BHT blue colormap)
CMAP = BHT_BLUE_CMAP

# ============================================================================


app = typer.Typer(pretty_exceptions_enable=False)


def load_evaluation_data(run_name: str, run_type: str = "aggregate"):
    """Load ground truth and predictions from a run.

    Args:
        run_name: Name of the run
        run_type: Type of run - "aggregate" or "infer"

    Returns:
        Tuple of (ground_truth, predictions) as lists of integers
    """
    io_name = "db_aggregate_to_json" if run_type == "aggregate" else "db_infer_to_json"
    reader = IOAdapterFactory.create_reader(io_name)
    eval_data = reader.read(run_name)

    # Convert RelevanceScore enums to integers
    ground_truth = [int(gt) for gt in eval_data.ground_truth]
    predictions = [
        int(pred) if pred is not None else -1 for pred in eval_data.predictions
    ]

    return ground_truth, predictions


def plot_confusion_matrix_figure(
    ground_truth: list,
    predictions: list,
    labels: list,
    output_path: Path,
    title: str = None,
    normalize: str = None,
    cmap: str = "Blues",
):
    """Create and save confusion matrix figure.

    Args:
        ground_truth: List of true labels (integers 0-3)
        predictions: List of predicted labels (integers 0-3)
        labels: Display labels for each class
        output_path: Where to save the figure
        title: Optional custom title
        normalize: Normalization mode (None, 'true', 'pred', 'all')
        cmap: Colormap name
    """
    # Filter out invalid predictions (-1)
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    gt_valid = [ground_truth[i] for i in valid_indices]
    pred_valid = [predictions[i] for i in valid_indices]

    skipped = len(ground_truth) - len(gt_valid)
    if skipped > 0:
        typer.echo(f"Warning: Skipped {skipped} samples with invalid predictions")

    # Compute confusion matrix
    cm = confusion_matrix(
        gt_valid, pred_valid, labels=[0, 1, 2, 3], normalize=normalize
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4))

    # Plot heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=1.0)
    cbar_label = "Proportion" if normalize else "Count"
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=FONTSIZE_SMALL)
    cbar.ax.tick_params(labelsize=FONTSIZE_SMALL)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate x labels for better fit
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations in each cell
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            text_color = "white" if value > thresh else "black"
            text = format(value, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=FONTSIZE,
            )

    # Labels and title
    ax.set_xlabel("LLM Prediction", fontweight="bold")
    ax.set_ylabel("Human Label (Ground Truth)", fontweight="bold")

    if title:
        ax.set_title(title, pad=10)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()

    # Copy to Overleaf
    copy_figure_to_overleaf(output_path)


def print_confusion_analysis(ground_truth: list, predictions: list):
    """Print analysis of confusion patterns using sklearn metrics.

    Args:
        ground_truth: List of true labels
        predictions: List of predicted labels
    """
    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    gt_valid = [ground_truth[i] for i in valid_indices]
    pred_valid = [predictions[i] for i in valid_indices]

    labels = [0, 1, 2, 3]
    class_names = ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]

    typer.echo("\n" + "=" * 50)
    typer.echo("Confusion Analysis")
    typer.echo("=" * 50)

    # Overall accuracy (sklearn)
    accuracy = accuracy_score(gt_valid, pred_valid)
    correct = int(accuracy * len(gt_valid))
    typer.echo(f"\nAccuracy: {accuracy:.1%} ({correct:,}/{len(gt_valid):,})")

    # Per-class recall (sklearn)
    recalls = recall_score(
        gt_valid, pred_valid, labels=labels, average=None, zero_division=0
    )
    typer.echo(
        "\nPer-class Recall (what % of each human label did LLM predict correctly):"
    )
    for i, (name, rec) in enumerate(zip(class_names, recalls)):
        support = sum(1 for g in gt_valid if g == i)
        typer.echo(f"  {name}: {rec:.1%} ({int(rec * support)}/{support})")

    # Per-class precision (sklearn)
    precisions = precision_score(
        gt_valid, pred_valid, labels=labels, average=None, zero_division=0
    )
    typer.echo("\nPer-class Precision (what % of LLM predictions were correct):")
    for i, (name, prec) in enumerate(zip(class_names, precisions)):
        predicted_count = sum(1 for p in pred_valid if p == i)
        typer.echo(
            f"  {name}: {prec:.1%} ({int(prec * predicted_count)}/{predicted_count})"
        )

    # Custom metrics: Over/under prediction analysis
    over_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p > g)
    under_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p < g)

    typer.echo(f"\nBias Analysis:")
    typer.echo(
        f"  Over-predicting (LLM > Human): {over_predict:,} ({over_predict/len(gt_valid):.1%})"
    )
    typer.echo(
        f"  Under-predicting (LLM < Human): {under_predict:,} ({under_predict/len(gt_valid):.1%})"
    )
    typer.echo(f"  Exact match: {correct:,} ({accuracy:.1%})")

    # Custom metrics: Adjacent vs distant errors
    adjacent_errors = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) == 1)
    distant_errors = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) >= 2)

    typer.echo(f"\nDisagreement:")
    typer.echo(
        f"  Soft Disagreement: {adjacent_errors:,} ({adjacent_errors/len(gt_valid):.1%})"
    )
    typer.echo(
        f"  Hard Disagreement: {distant_errors:,} ({distant_errors/len(gt_valid):.1%})"
    )


def save_confusion_matrix_table(
    ground_truth: list, predictions: list, output_path: Path
):
    """Save confusion matrix and analysis as a formatted text table using sklearn metrics.

    Args:
        ground_truth: List of true labels
        predictions: List of predicted labels
        output_path: Where to save the table
    """
    # Filter valid
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    gt_valid = [ground_truth[i] for i in valid_indices]
    pred_valid = [predictions[i] for i in valid_indices]

    labels = [0, 1, 2, 3]
    class_names = ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]
    cm = confusion_matrix(gt_valid, pred_valid, labels=labels)

    lines = []
    lines.append("Confusion Matrix (rows=Human/Ground Truth, cols=LLM Prediction)")
    lines.append("")
    lines.append("                    LLM Prediction")
    lines.append("                    0       1       2       3     | Row Total")
    lines.append(
        "Human  0 (Irrel)  {:5d}   {:5d}   {:5d}   {:5d}   | {:5d}".format(
            cm[0, 0], cm[0, 1], cm[0, 2], cm[0, 3], cm[0].sum()
        )
    )
    lines.append(
        "       1 (Relev)  {:5d}   {:5d}   {:5d}   {:5d}   | {:5d}".format(
            cm[1, 0], cm[1, 1], cm[1, 2], cm[1, 3], cm[1].sum()
        )
    )
    lines.append(
        "       2 (High)   {:5d}   {:5d}   {:5d}   {:5d}   | {:5d}".format(
            cm[2, 0], cm[2, 1], cm[2, 2], cm[2, 3], cm[2].sum()
        )
    )
    lines.append(
        "       3 (Perf)   {:5d}   {:5d}   {:5d}   {:5d}   | {:5d}".format(
            cm[3, 0], cm[3, 1], cm[3, 2], cm[3, 3], cm[3].sum()
        )
    )
    lines.append("       " + "-" * 50)
    lines.append(
        "Col Total         {:5d}   {:5d}   {:5d}   {:5d}   | {:5d}".format(
            cm[:, 0].sum(), cm[:, 1].sum(), cm[:, 2].sum(), cm[:, 3].sum(), cm.sum()
        )
    )

    # Add analysis
    lines.append("")
    lines.append("=" * 50)
    lines.append("Analysis")
    lines.append("=" * 50)

    # Overall accuracy (sklearn)
    accuracy = accuracy_score(gt_valid, pred_valid)
    correct = int(accuracy * len(gt_valid))
    lines.append("")
    lines.append(f"Accuracy: {accuracy:.1%} ({correct:,}/{len(gt_valid):,})")

    # Per-class recall (sklearn)
    recalls = recall_score(
        gt_valid, pred_valid, labels=labels, average=None, zero_division=0
    )
    lines.append("")
    lines.append(
        "Per-class Recall (what % of each human label did LLM predict correctly):"
    )
    for i, (name, rec) in enumerate(zip(class_names, recalls)):
        support = sum(1 for g in gt_valid if g == i)
        lines.append(f"  {name}: {rec:.1%} ({int(rec * support)}/{support})")

    # Per-class precision (sklearn)
    precisions = precision_score(
        gt_valid, pred_valid, labels=labels, average=None, zero_division=0
    )
    lines.append("")
    lines.append("Per-class Precision (what % of LLM predictions were correct):")
    for i, (name, prec) in enumerate(zip(class_names, precisions)):
        predicted_count = sum(1 for p in pred_valid if p == i)
        lines.append(
            f"  {name}: {prec:.1%} ({int(prec * predicted_count)}/{predicted_count})"
        )

    # Custom metrics: Bias analysis
    over_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p > g)
    under_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p < g)

    lines.append("")
    lines.append("Bias Analysis:")
    lines.append(
        f"  Over-predicting (LLM > Human): {over_predict:,} ({over_predict/len(gt_valid):.1%})"
    )
    lines.append(
        f"  Under-predicting (LLM < Human): {under_predict:,} ({under_predict/len(gt_valid):.1%})"
    )
    lines.append(f"  Exact match: {correct:,} ({accuracy:.1%})")

    # Custom metrics: Error distance
    adjacent_errors = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) == 1)
    distant_errors = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) >= 2)

    lines.append("")
    lines.append("Disagreement:")
    lines.append(
        f"  Soft Disagreement: {adjacent_errors:,} ({adjacent_errors/len(gt_valid):.1%})"
    )
    lines.append(
        f"  Hard Disagreement: {distant_errors:,} ({distant_errors/len(gt_valid):.1%})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    typer.echo(f"Table saved to: {output_path}")


def save_confusion_matrix_latex(
    ground_truth: list, predictions: list, output_path: Path
):
    """Save confusion matrix as a LaTeX table using sklearn metrics.

    Args:
        ground_truth: List of true labels
        predictions: List of predicted labels
        output_path: Where to save the LaTeX table
    """
    # Filter valid
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    gt_valid = [ground_truth[i] for i in valid_indices]
    pred_valid = [predictions[i] for i in valid_indices]

    labels = [0, 1, 2, 3]
    class_names = ["Irrelevant", "Relevant", "Highly Relevant", "Perfectly Relevant"]
    cm = confusion_matrix(gt_valid, pred_valid, labels=labels)

    # Calculate metrics using sklearn
    accuracy = accuracy_score(gt_valid, pred_valid)
    recalls = recall_score(
        gt_valid, pred_valid, labels=labels, average=None, zero_division=0
    )
    precisions = precision_score(
        gt_valid, pred_valid, labels=labels, average=None, zero_division=0
    )

    # Custom metrics
    over_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p > g)
    under_predict = sum(1 for g, p in zip(gt_valid, pred_valid) if p < g)
    adjacent_errors = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) == 1)
    distant_errors = sum(1 for g, p in zip(gt_valid, pred_valid) if abs(g - p) >= 2)

    lines = []

    # Confusion matrix table
    lines.append("% Confusion Matrix")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Confusion Matrix (rows = Human Labels, columns = LLM Predictions)}"
    )
    lines.append("\\label{tab:confusion-matrix}")
    lines.append("\\begin{tabular}{l|rrrr|r}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{4}{c|}{LLM Prediction} & \\\\")
    lines.append("Human Label & 0 & 1 & 2 & 3 & Total \\\\")
    lines.append("\\midrule")

    row_labels = [
        "0 (Irrelevant)",
        "1 (Relevant)",
        "2 (Highly Rel.)",
        "3 (Perfectly Rel.)",
    ]
    for i, label in enumerate(row_labels):
        row_values = " & ".join(str(cm[i, j]) for j in range(4))
        lines.append(f"{label} & {row_values} & {cm[i].sum()} \\\\")

    lines.append("\\midrule")
    col_totals = " & ".join(str(cm[:, j].sum()) for j in range(4))
    lines.append(f"Total & {col_totals} & {cm.sum()} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    lines.append("")
    lines.append("")

    # Analysis metrics table
    lines.append("% Analysis Metrics")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Confusion Matrix Analysis}")
    lines.append("\\label{tab:confusion-analysis}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Metric & Value \\\\")
    lines.append("\\midrule")
    lines.append(f"Accuracy & {accuracy:.1%}".replace("%", "\\%") + " \\\\")
    lines.append("\\midrule")

    # Per-class recall (sklearn)
    for name, rec in zip(class_names, recalls):
        lines.append(f"Recall ({name}) & {rec:.1%}".replace("%", "\\%") + " \\\\")

    lines.append("\\midrule")

    # Per-class precision (sklearn)
    for name, prec in zip(class_names, precisions):
        lines.append(f"Precision ({name}) & {prec:.1%}".replace("%", "\\%") + " \\\\")

    lines.append("\\midrule")
    lines.append(
        f"Over-predicting (LLM > Human) & {over_predict/len(gt_valid):.1%}".replace(
            "%", "\\%"
        )
        + " \\\\"
    )
    lines.append(
        f"Under-predicting (LLM < Human) & {under_predict/len(gt_valid):.1%}".replace(
            "%", "\\%"
        )
        + " \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        f"Soft Disagreement & {adjacent_errors/len(gt_valid):.1%}".replace("%", "\\%")
        + " \\\\"
    )
    lines.append(
        f"Hard Disagreement & {distant_errors/len(gt_valid):.1%}".replace("%", "\\%")
        + " \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    typer.echo(f"LaTeX table saved to: {output_path}")


@app.command()
def main(
    run_name: str = typer.Option(RUN_NAME, "--run-name", "-r", help="Run name"),
    run_type: str = typer.Option(
        RUN_TYPE, "--run-type", "-t", help="Run type: aggregate or infer"
    ),
    normalize: str = typer.Option(
        NORMALIZE, "--normalize", "-n", help="Normalization: none, true, pred, all"
    ),
):
    """Plot confusion matrix for LLM vs human relevance labels.

    Configuration is set via constants at the top of this file.
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = project_root / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames
    safe_run_name = run_name.replace("/", "_").replace("\\", "_")
    norm_suffix = f"_{normalize}" if normalize else ""

    output_filename = OUTPUT_FILENAME
    if output_filename is None:
        output_filename = f"confusion_matrix{norm_suffix}_{safe_run_name}.svg"
    output_path = figures_dir / output_filename

    table_filename = f"confusion_matrix{norm_suffix}_{safe_run_name}.txt"
    table_path = tables_dir / table_filename

    latex_filename = f"confusion_matrix{norm_suffix}_{safe_run_name}.tex"
    latex_path = tables_dir / latex_filename

    typer.echo(f"Loading data from {run_type} run: {run_name}")

    # Load data
    ground_truth, predictions = load_evaluation_data(run_name, run_type)
    typer.echo(f"Loaded {len(ground_truth):,} samples")

    # Print analysis
    print_confusion_analysis(ground_truth, predictions)

    # Save tables
    save_confusion_matrix_table(ground_truth, predictions, table_path)
    save_confusion_matrix_latex(ground_truth, predictions, latex_path)

    # Create plot
    plot_confusion_matrix_figure(
        ground_truth=ground_truth,
        predictions=predictions,
        labels=RELEVANCE_LABELS_SHORT,
        output_path=output_path,
        title=PLOT_TITLE,
        normalize=normalize if normalize != "none" else None,
        cmap=CMAP,
    )


if __name__ == "__main__":
    app()
