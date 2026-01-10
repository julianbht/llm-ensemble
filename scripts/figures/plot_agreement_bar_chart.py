#!/usr/bin/env python3
"""Plot agreement metric bar chart comparing individual models vs ensemble.

This script reads evaluate_run.json files from multiple runs and creates
a bar chart comparing their performance on a specified metric (e.g., Cohen's kappa).

Usage:
    # Run with defaults from constants below
    python scripts/figures/plot_agreement_bar_chart.py

    # Override specific settings
    python scripts/figures/plot_agreement_bar_chart.py --metric krippendorffs_alpha
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import typer


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================

# List of (label, run_name) tuples to compare
RUNS = [
    ("llama-3.2-3b-instruct", "ensemble-4-meta-llama-3.2-3b-instruct-start"),
    ("ministral-3b-2515", "ensemble-2-ministral-3b-2515"),
    ("google-gemma-3-4b-it", "ensemble-1-google-gemma-3-4b-it"),
    ("phi-4-multimodal-instruct", "ensemble-3-phi-4-multimodal-instruct"),
    ("ui-tars-1.5-7b", "ensemble-5-ui-tars-1.5-7b-start"),
    ("ensemble (avg)", "ensemble-aggregated-avg-vote-1-to-5"),
    # ("cohere-command-r7b-12-2024", "ensemble-6-cohere-command-r7b-12-2024-start"),
]

# Metric to plot (must exist in evaluate_run.json metric_results)
METRIC_NAME = "cohens_kappa"

# Run type directory (subdirectory under artifacts/runs/evaluate/)
RUN_TYPE = "official"

# Y-axis limits: None for auto-scaling, or [min, max] to fix the range
# Examples: [0.0, 1.0], [0.0, 0.75], [-1.0, 1.0], None
Y_AXIS_LIMITS = [0.0, 1.0]

# Output filename (None = auto-generate from metric name)
OUTPUT_FILENAME = None  # e.g., "my_figure.svg" or None

# Custom plot title (None = auto-generate)
PLOT_TITLE = None  # e.g., "Individual vs Ensemble Agreement"

# ============================================================================


app = typer.Typer()


def read_evaluate_run(run_path: Path) -> dict:
    """Read and parse evaluate_run.json file."""
    with open(run_path / "evaluate_run.json", "r") as f:
        return json.load(f)


def extract_metric_value(evaluate_run: dict, metric_name: str) -> float:
    """Extract specified metric value from metric results.

    Args:
        evaluate_run: Parsed evaluate_run.json dict
        metric_name: Name of metric to extract (e.g., 'cohens_kappa')

    Returns:
        Metric value

    Raises:
        ValueError: If metric not found
    """
    for metric in evaluate_run["metric_results"]:
        if metric["name"] == metric_name:
            return metric["value"]
    raise ValueError(f"Metric '{metric_name}' not found in metric results")


def collect_data(
    runs: List[Tuple[str, str]], evaluate_runs_base: Path, metric_name: str
) -> Tuple[List[str], List[float]]:
    """Collect (label, metric_value) pairs from evaluate runs.

    Args:
        runs: List of (label, run_name) tuples
        evaluate_runs_base: Base path for evaluate runs
        metric_name: Name of metric to extract

    Returns:
        Tuple of (labels, values)
    """
    labels = []
    values = []

    for label, run_name in runs:
        run_path = evaluate_runs_base / run_name
        if not run_path.exists():
            typer.echo(f"Warning: Run path does not exist: {run_path}", err=True)
            continue

        evaluate_run = read_evaluate_run(run_path)

        try:
            metric_value = extract_metric_value(evaluate_run, metric_name)
            labels.append(label)
            values.append(metric_value)
            typer.echo(f"Found: {label} = {metric_value:.4f} ({run_name})")
        except ValueError as e:
            typer.echo(f"Warning: {e} for run {run_name}", err=True)
            continue

    return labels, values


def plot_agreement_bar_chart(
    labels: List[str],
    values: List[float],
    metric_name: str,
    output_path: Path,
    y_limits: Optional[List[float]] = None,
    title: Optional[str] = None,
):
    """Create and save bar chart of agreement metrics.

    Args:
        labels: Model/run labels for x-axis
        values: Metric values for each model
        metric_name: Name of the metric (for axis label)
        output_path: Where to save the figure
        y_limits: Y-axis limits [min, max] or None for auto-scaling
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(
        labels, values, color="steelblue", alpha=0.8, edgecolor="black", linewidth=1.2
    )

    # Highlight the last bar (assumed to be ensemble) in different color
    if len(bars) > 1:
        bars[-1].set_color("coral")
        bars[-1].set_alpha(0.9)

    # Set y-axis limits if specified
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    # Labels and title
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")

    metric_display = metric_name.replace("_", " ").title()
    ax.set_ylabel(f"{metric_display}", fontsize=12, fontweight="bold")

    if title is None:
        title = f"Agreement Metrics: {metric_display}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Grid for readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Rotate x-labels if many models
    if len(labels) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()


@app.command()
def main():
    """Plot agreement metric bar chart comparing models.

    Configuration is set via constants at the top of this file.

    Example:
        python scripts/figures/plot_agreement_bar_chart.py
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    evaluate_runs_base = project_root / "artifacts" / "runs" / "evaluate" / RUN_TYPE
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_filename = OUTPUT_FILENAME
    if output_filename is None:
        output_filename = f"agreement_bar_chart_{METRIC_NAME}.svg"
    output_path = figures_dir / output_filename

    typer.echo(f"Analyzing {len(RUNS)} evaluate runs from: {evaluate_runs_base}")
    typer.echo(f"Metric: {METRIC_NAME}\n")

    # Collect data
    labels, values = collect_data(RUNS, evaluate_runs_base, METRIC_NAME)

    if not labels:
        typer.echo(
            "Error: No data collected. Check that run paths exist and contain valid data.",
            err=True,
        )
        raise typer.Exit(1)

    # Create plot
    plot_agreement_bar_chart(
        labels, values, METRIC_NAME, output_path, Y_AXIS_LIMITS, PLOT_TITLE
    )


if __name__ == "__main__":
    app()
