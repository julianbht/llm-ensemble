#!/usr/bin/env python3
"""Plot ensemble size vs Cohen's kappa from evaluate runs.

This script reads evaluate_run.json files from multiple runs and creates
a plot showing how Cohen's kappa varies with ensemble size.

Usage:
    # Run with defaults from constants below
    python scripts/figures/plot_ensemble_size_vs_kappa.py
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import typer


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================

# List of (ensemble_size, run_name) tuples to plot
# The ensemble_size is explicit rather than extracted from the run name
RUNS = [
    (1, "1-ensemble-size-analysis"),
    (2, "2-ensemble-size-analysis"),
    (3, "3-ensemble-size-analysis"),
    (4, "4-ensemble-size-analysis"),
    (5, "5-ensemble-size-analysis"),
    (6, "6-ensemble-size-analysis"),
    (7, "7-ensemble-size-analysis"),
]

# Metric to plot (must exist in evaluate_run.json metric_results)
METRIC_NAME = "cohens_kappa"

# Run type directory (subdirectory under artifacts/runs/evaluate/)
RUN_TYPE = "official"

# Y-axis limits: None for auto-scaling, or [min, max] to fix the range
# Examples: [0.0, 1.0], [0.0, 0.75], [-1.0, 1.0], None
Y_AXIS_LIMITS = None

# Output filename (None = auto-generate from metric name)
OUTPUT_FILENAME = None  # e.g., "ensemble_size_vs_kappa.svg" or None

# Custom plot title (None = auto-generate)
PLOT_TITLE = None  # e.g., "Ensemble Size vs Cohen's Kappa"

# Output format: "svg" or "png"
OUTPUT_FORMAT = "svg"

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
    runs: List[Tuple[int, str]], evaluate_runs_base: Path, metric_name: str
) -> List[Tuple[int, float]]:
    """Collect (ensemble_size, metric_value) pairs from evaluate runs.

    Args:
        runs: List of (ensemble_size, run_name) tuples
        evaluate_runs_base: Base path for evaluate runs
        metric_name: Name of metric to extract

    Returns:
        List of (ensemble_size, metric_value) tuples sorted by ensemble size
    """
    data = []

    for ensemble_size, run_name in runs:
        run_path = evaluate_runs_base / run_name
        if not run_path.exists():
            typer.echo(f"Warning: Run path does not exist: {run_path}", err=True)
            continue

        evaluate_run = read_evaluate_run(run_path)

        try:
            metric_value = extract_metric_value(evaluate_run, metric_name)
            data.append((ensemble_size, metric_value))
            typer.echo(
                f"Found: ensemble_size={ensemble_size}, {metric_name}={metric_value:.4f} ({run_name})"
            )
        except ValueError as e:
            typer.echo(f"Warning: {e} for run {run_name}", err=True)
            continue

    return sorted(data, key=lambda x: x[0])


def plot_ensemble_size_vs_metric(
    data: List[Tuple[int, float]],
    metric_name: str,
    output_path: Path,
    y_limits: Optional[List[float]] = None,
    title: Optional[str] = None,
):
    """Create and save plot of ensemble size vs metric.

    Args:
        data: List of (ensemble_size, metric_value) tuples
        metric_name: Name of the metric (for axis label)
        output_path: Where to save the figure
        y_limits: Y-axis limits [min, max] or None for auto-scaling
        title: Optional custom title
    """
    ensemble_sizes = [d[0] for d in data]
    metric_values = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line with markers
    ax.plot(
        ensemble_sizes,
        metric_values,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="steelblue",
        markerfacecolor="coral",
        markeredgecolor="black",
        markeredgewidth=1.2,
    )

    # Set y-axis limits if specified
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    # Labels and title
    ax.set_xlabel("Ensemble Size", fontsize=12, fontweight="bold")

    metric_display = metric_name.replace("_", " ").title()
    ax.set_ylabel(f"{metric_display}", fontsize=12, fontweight="bold")

    if title is None:
        title = f"Ensemble Size vs {metric_display}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Show all ensemble sizes on x-axis
    ax.set_xticks(ensemble_sizes)

    # Add value labels on points
    for size, value in data:
        ax.annotate(
            f"{value:.3f}",
            xy=(size, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format=OUTPUT_FORMAT)
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()


@app.command()
def main():
    """Plot ensemble size vs metric from evaluate runs.

    Configuration is set via constants at the top of this file.

    Example:
        python scripts/figures/plot_ensemble_size_vs_kappa.py
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    evaluate_runs_base = project_root / "artifacts" / "runs" / "evaluate" / RUN_TYPE
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_filename = OUTPUT_FILENAME
    if output_filename is None:
        output_filename = f"ensemble_size_vs_{METRIC_NAME}.{OUTPUT_FORMAT}"
    output_path = figures_dir / output_filename

    typer.echo(f"Analyzing {len(RUNS)} evaluate runs from: {evaluate_runs_base}")
    typer.echo(f"Metric: {METRIC_NAME}\n")

    # Collect data
    data = collect_data(RUNS, evaluate_runs_base, METRIC_NAME)

    if not data:
        typer.echo(
            "Error: No data collected. Check that run paths exist and contain valid data.",
            err=True,
        )
        raise typer.Exit(1)

    # Create plot
    plot_ensemble_size_vs_metric(
        data, METRIC_NAME, output_path, Y_AXIS_LIMITS, PLOT_TITLE
    )


if __name__ == "__main__":
    app()
