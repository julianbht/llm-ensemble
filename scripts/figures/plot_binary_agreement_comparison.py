#!/usr/bin/env python3
"""Plot grouped bar chart comparing graded vs binary Cohen's kappa.

This script demonstrates that low agreement on graded relevance (0-3) improves
substantially when binarized (relevant vs not relevant), suggesting disagreement
stems from fine-grained distinctions rather than fundamental relevance disagreement.

Binarization strategy:
- 0 = Not Relevant
- 1, 2, 3 = Relevant

Usage:
    python scripts/figures/plot_binary_agreement_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import typer
from sklearn.metrics import cohen_kappa_score

from thesis_colors import BHT_COLORS, GRAY_SCALE
from copy_to_overleaf import copy_figure_to_overleaf

from llm_ensemble.evaluate.startup import dependency_configurator  # noqa: F401
from llm_ensemble.evaluate.adapters.driven.io_factory import IOAdapterFactory


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================

# List of (label, run_name, run_type) tuples to compare
RUNS = [
    ("llama-3.2-3b", "ensemble-4-meta-llama-3.2-3b-instruct-start", "infer"),
    ("ministral-3b", "ensemble-2-ministral-3b-2515", "infer"),
    ("gemma-3-4b", "ensemble-1-google-gemma-3-4b-it", "infer"),
    ("phi-4", "ensemble-3-phi-4-multimodal-instruct", "infer"),
    ("ui-tars-1.5-7b", "ensemble-5-ui-tars-1.5-7b-start", "infer"),
    ("Ensemble (MVA)", "-ensemble-1-to-5-majority-vote-average", "aggregate"),
    ("Ensemble (MVR)", "-ensemble-1-to-5-majority-vote-random", "aggregate"),
    ("GPT-5.1", "reference-ensemble-gpt-5-1-all-samples-start", "infer"),
]

# Binarization threshold: labels >= threshold are "relevant"
# With threshold=1: 0=not relevant, 1/2/3=relevant
BINARY_THRESHOLD = 1

# Y-axis limits
Y_AXIS_LIMITS = [0.0, 0.6]

# Output filename (None = auto-generate)
OUTPUT_FILENAME = "binary_agreement_comparison-1-limit.svg"

# Plot title
PLOT_TITLE = "Graded vs Binary Agreement (Cohen's κ)"

# ============================================================================


app = typer.Typer()


def load_evaluation_data(run_name: str, run_type: str) -> Tuple[list[int], list[int]]:
    """Load ground truth and predictions from a run."""
    io_name = "db_aggregate_to_json" if run_type == "aggregate" else "db_infer_to_json"
    reader = IOAdapterFactory.create_reader(io_name)
    eval_data = reader.read(run_name)

    ground_truth = [int(gt) for gt in eval_data.ground_truth]
    predictions = [
        int(pred) if pred is not None else -1 for pred in eval_data.predictions
    ]

    return ground_truth, predictions


def binarize_labels(labels: list[int], threshold: int) -> list[int]:
    """Binarize labels: 0 if < threshold, 1 if >= threshold."""
    return [1 if label >= threshold else 0 for label in labels]


def compute_kappas(
    ground_truth: list[int], predictions: list[int], threshold: int
) -> Tuple[float, float]:
    """Compute graded and binary Cohen's kappa.

    Returns:
        Tuple of (graded_kappa, binary_kappa)
    """
    # Filter invalid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    gt_valid = [ground_truth[i] for i in valid_indices]
    pred_valid = [predictions[i] for i in valid_indices]

    # Graded kappa (0-3 scale)
    graded_kappa = cohen_kappa_score(gt_valid, pred_valid)

    # Binary kappa
    gt_binary = binarize_labels(gt_valid, threshold)
    pred_binary = binarize_labels(pred_valid, threshold)
    binary_kappa = cohen_kappa_score(gt_binary, pred_binary)

    return graded_kappa, binary_kappa


def plot_grouped_bar_chart(
    labels: list[str],
    graded_kappas: list[float],
    binary_kappas: list[float],
    output_path: Path,
    y_limits: list[float] = None,
    title: str = None,
):
    """Create grouped bar chart comparing graded vs binary kappa."""
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped bars
    bars_graded = ax.bar(
        x - width / 2,
        graded_kappas,
        width,
        label="Graded (0-3)",
        color=BHT_COLORS["turquoise"],
        edgecolor="black",
        linewidth=0.5,
    )
    bars_binary = ax.bar(
        x + width / 2,
        binary_kappas,
        width,
        label="Binary (relevant/not)",
        color=BHT_COLORS["blue"],
        edgecolor="black",
        linewidth=0.5,
    )

    # Y-axis limits
    if y_limits:
        ax.set_ylim(y_limits[0], y_limits[1])

    # Labels and title
    ax.set_xlabel("Model / Ensemble", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cohen's κ", fontsize=12, fontweight="bold")
    ax.set_title(title or "Graded vs Binary Agreement", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Add value labels on bars
    for bars in [bars_graded, bars_binary]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()

    copy_figure_to_overleaf(output_path)


def save_comparison_table(
    labels: list[str],
    graded_kappas: list[float],
    binary_kappas: list[float],
    output_path: Path,
):
    """Save comparison as text table."""
    lines = []
    lines.append("Graded vs Binary Cohen's Kappa Comparison")
    lines.append("=" * 60)
    lines.append(f"Binarization: 0 = not relevant, >= {BINARY_THRESHOLD} = relevant")
    lines.append("")
    lines.append(f"{'Model':<25} {'Graded κ':>12} {'Binary κ':>12} {'Δ':>10}")
    lines.append("-" * 60)

    for label, graded, binary in zip(labels, graded_kappas, binary_kappas):
        delta = binary - graded
        lines.append(f"{label:<25} {graded:>12.3f} {binary:>12.3f} {delta:>+10.3f}")

    lines.append("-" * 60)
    avg_graded = np.mean(graded_kappas)
    avg_binary = np.mean(binary_kappas)
    avg_delta = avg_binary - avg_graded
    lines.append(
        f"{'Average':<25} {avg_graded:>12.3f} {avg_binary:>12.3f} {avg_delta:>+10.3f}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    typer.echo(f"Table saved to: {output_path}")


def save_comparison_latex(
    labels: list[str],
    graded_kappas: list[float],
    binary_kappas: list[float],
    output_path: Path,
):
    """Save comparison as LaTeX table."""
    lines = []
    lines.append("% Graded vs Binary Kappa Comparison")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Graded vs Binary Agreement (threshold: $\\geq {BINARY_THRESHOLD}$ = relevant)}}"
    )
    lines.append("\\label{tab:binary-agreement-comparison}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Model & Graded $\\kappa$ & Binary $\\kappa$ & $\\Delta$ \\\\")
    lines.append("\\midrule")

    for label, graded, binary in zip(labels, graded_kappas, binary_kappas):
        delta = binary - graded
        # Escape underscores in labels
        safe_label = label.replace("_", "\\_")
        lines.append(f"{safe_label} & {graded:.3f} & {binary:.3f} & {delta:+.3f} \\\\")

    lines.append("\\midrule")
    avg_graded = np.mean(graded_kappas)
    avg_binary = np.mean(binary_kappas)
    avg_delta = avg_binary - avg_graded
    lines.append(
        f"Average & {avg_graded:.3f} & {avg_binary:.3f} & {avg_delta:+.3f} \\\\"
    )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    typer.echo(f"LaTeX table saved to: {output_path}")


@app.command()
def main(
    threshold: int = typer.Option(
        BINARY_THRESHOLD, "--threshold", "-t", help="Binarization threshold"
    ),
):
    """Compare graded vs binary Cohen's kappa across runs."""
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = project_root / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / OUTPUT_FILENAME
    table_path = tables_dir / "binary_agreement_comparison.txt"
    latex_path = tables_dir / "binary_agreement_comparison.tex"

    typer.echo(f"Binarization threshold: >= {threshold} is relevant")
    typer.echo(f"Analyzing {len(RUNS)} runs...\n")

    labels = []
    graded_kappas = []
    binary_kappas = []

    for label, run_name, run_type in RUNS:
        try:
            gt, pred = load_evaluation_data(run_name, run_type)
            graded_k, binary_k = compute_kappas(gt, pred, threshold)

            labels.append(label)
            graded_kappas.append(graded_k)
            binary_kappas.append(binary_k)

            delta = binary_k - graded_k
            typer.echo(
                f"{label}: graded={graded_k:.3f}, binary={binary_k:.3f} (Δ={delta:+.3f})"
            )
        except Exception as e:
            typer.echo(f"Warning: Failed to load {run_name}: {e}", err=True)

    if not labels:
        typer.echo("Error: No data loaded", err=True)
        raise typer.Exit(1)

    # Summary
    avg_improvement = np.mean([b - g for g, b in zip(graded_kappas, binary_kappas)])
    typer.echo(f"\nAverage improvement: {avg_improvement:+.3f}")

    # Save outputs
    save_comparison_table(labels, graded_kappas, binary_kappas, table_path)
    save_comparison_latex(labels, graded_kappas, binary_kappas, latex_path)
    plot_grouped_bar_chart(
        labels, graded_kappas, binary_kappas, output_path, Y_AXIS_LIMITS, PLOT_TITLE
    )


if __name__ == "__main__":
    app()
