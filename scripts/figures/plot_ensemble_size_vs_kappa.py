#!/usr/bin/env python3
"""Plot ensemble size vs metric for multiple aggregation strategies.

This script reads evaluate_run.json files from multiple runs and creates
a plot showing how a metric varies with ensemble size across different
aggregation strategies.

Usage:
    # Run with defaults from constants below
    python scripts/figures/plot_ensemble_size_vs_kappa.py
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import typer

from thesis_colors import (
    BHT_COLORS,
    FIGURE_WIDTH,
    apply_thesis_style,
)
from copy_to_overleaf import copy_figure_to_overleaf

apply_thesis_style()


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================

# Dictionary mapping strategy name to list of (ensemble_size, run_name) tuples
# The ensemble_size is explicit rather than extracted from the run name
RUNS_BY_STRATEGY = {
    "average_vote": [
        (1, "1-ensemble-size-analysis-average_vote-params_small_to_big"),
        (2, "2-ensemble-size-analysis-average_vote-params_small_to_big"),
        (3, "3-ensemble-size-analysis-average_vote-params_small_to_big"),
        (4, "4-ensemble-size-analysis-average_vote-params_small_to_big"),
        (5, "5-ensemble-size-analysis-average_vote-params_small_to_big"),
        (6, "6-ensemble-size-analysis-average_vote-params_small_to_big"),
        (7, "7-ensemble-size-analysis-average_vote-params_small_to_big"),
    ],
    "majority_vote_average": [
        (1, "1-ensemble-size-analysis-majority_vote_average-params_small_to_big"),
        (2, "2-ensemble-size-analysis-majority_vote_average-params_small_to_big"),
        (3, "3-ensemble-size-analysis-majority_vote_average-params_small_to_big"),
        (4, "4-ensemble-size-analysis-majority_vote_average-params_small_to_big"),
        (5, "5-ensemble-size-analysis-majority_vote_average-params_small_to_big"),
        (6, "6-ensemble-size-analysis-majority_vote_average-params_small_to_big"),
        (7, "7-ensemble-size-analysis-majority_vote_average-params_small_to_big"),
    ],
    "majority_vote_random": [
        (1, "1-ensemble-size-analysis-majority_vote_random-params_small_to_big"),
        (2, "2-ensemble-size-analysis-majority_vote_random-params_small_to_big"),
        (3, "3-ensemble-size-analysis-majority_vote_random-params_small_to_big"),
        (4, "4-ensemble-size-analysis-majority_vote_random-params_small_to_big"),
        (5, "5-ensemble-size-analysis-majority_vote_random-params_small_to_big"),
        (6, "6-ensemble-size-analysis-majority_vote_random-params_small_to_big"),
        (7, "7-ensemble-size-analysis-majority_vote_random-params_small_to_big"),
    ],
}

# Metric to plot (must exist in evaluate_run.json metric_results)
METRIC_NAME = "cohens_kappa"

# Run type directory (subdirectory under artifacts/runs/evaluate/)
RUN_TYPE = "official"

# Y-axis limits: None for auto-scaling, or [min, max] to fix the range
# Examples: [0.0, 1.0], [0.0, 0.75], [-1.0, 1.0], None
Y_AXIS_LIMITS = None

# Output filename (None = auto-generate from metric name)
OUTPUT_FILENAME = "ensemble_size_vs_kappa_multi_strategy_params_small_to_big.svg"  # e.g., "ensemble_size_vs_kappa_multi_strategy.svg" or None

# Custom plot title (None = auto-generate)
PLOT_TITLE = "Ensemble Size vs Cohen's Kappa (small to big)"  # e.g., "Ensemble Size vs Cohen's Kappa (Multiple Strategies)"
# PLOT_TITLE = "Ensemble Size vs Cohen's Kappa (big to small)"  # e.g., "Ensemble Size vs Cohen's Kappa (Multiple Strategies)"

# Output format: "svg" or "png"
OUTPUT_FORMAT = "svg"

# Style configuration for each strategy (using BHT colors)
STRATEGY_STYLES = {
    "average_vote": {
        "color": BHT_COLORS["blue"],  # Dark blue
        "marker": "o",
        "label": "Average Vote",
    },
    "majority_vote_average": {
        "color": BHT_COLORS["red"],
        "marker": "s",
        "label": "Majority Vote Average",
    },
    "majority_vote_random": {
        "color": BHT_COLORS["turquoise"],
        "marker": "^",
        "label": "Majority Vote Random",
    },
}

# ============================================================================


app = typer.Typer(pretty_exceptions_enable=False)


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


def collect_data_by_strategy(
    runs_by_strategy: Dict[str, List[Tuple[int, str]]],
    evaluate_runs_base: Path,
    metric_name: str,
) -> Dict[str, List[Tuple[int, float]]]:
    """Collect metric data grouped by strategy.

    Args:
        runs_by_strategy: Dict mapping strategy to list of (ensemble_size, run_name)
        evaluate_runs_base: Base path for evaluate runs
        metric_name: Name of metric to extract

    Returns:
        Dict mapping strategy to list of (ensemble_size, metric_value) tuples
    """
    data_by_strategy = {}

    for strategy, runs in runs_by_strategy.items():
        typer.echo(f"\nCollecting data for strategy: {strategy}")
        data = []

        for ensemble_size, run_name in runs:
            run_path = evaluate_runs_base / run_name
            if not run_path.exists():
                typer.echo(f"  Warning: Run path does not exist: {run_path}", err=True)
                continue

            evaluate_run = read_evaluate_run(run_path)

            try:
                metric_value = extract_metric_value(evaluate_run, metric_name)
                data.append((ensemble_size, metric_value))
                typer.echo(
                    f"  Found: ensemble_size={ensemble_size}, {metric_name}={metric_value:.4f}"
                )
            except ValueError as e:
                typer.echo(f"  Warning: {e} for run {run_name}", err=True)
                continue

        data_by_strategy[strategy] = sorted(data, key=lambda x: x[0])

    return data_by_strategy


def plot_ensemble_size_vs_metric_multi_strategy(
    data_by_strategy: Dict[str, List[Tuple[int, float]]],
    metric_name: str,
    output_path: Path,
    y_limits: Optional[List[float]] = None,
    title: Optional[str] = None,
):
    """Create and save plot of ensemble size vs metric for multiple strategies.

    Args:
        data_by_strategy: Dict mapping strategy to list of (size, value) tuples
        metric_name: Name of the metric (for axis label)
        output_path: Where to save the figure
        y_limits: Y-axis limits [min, max] or None for auto-scaling
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4))

    # Plot each strategy
    for strategy, data in data_by_strategy.items():
        if not data:
            typer.echo(f"Warning: No data for strategy {strategy}, skipping", err=True)
            continue

        ensemble_sizes = [d[0] for d in data]
        metric_values = [d[1] for d in data]

        style = STRATEGY_STYLES.get(
            strategy,
            {"color": "gray", "marker": "x", "label": strategy},
        )

        ax.plot(
            ensemble_sizes,
            metric_values,
            marker=style["marker"],
            linewidth=1.5,
            markersize=6,
            color=style["color"],
            label=style["label"],
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    # Set y-axis limits if specified
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    # Labels and title
    ax.set_xlabel("Ensemble Size", fontweight="bold")

    metric_display = metric_name.replace("_", " ").title()
    ax.set_ylabel(f"{metric_display}", fontweight="bold")

    if title is None:
        title = f"Ensemble Size vs {metric_display}"
    ax.set_title(title, pad=20)

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Show all ensemble sizes on x-axis
    if data_by_strategy:
        all_sizes = sorted(
            set(size for data in data_by_strategy.values() for size, _ in data)
        )
        ax.set_xticks(all_sizes)

    # Legend
    ax.legend(loc="best", framealpha=0.95)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", format=OUTPUT_FORMAT)
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()

    # Copy to Overleaf
    copy_figure_to_overleaf(output_path)


def save_data_table(
    data_by_strategy: Dict[str, List[Tuple[int, float]]],
    metric_name: str,
    output_path: Path,
):
    """Save the data as a CSV table.

    Args:
        data_by_strategy: Dict mapping strategy to list of (size, value) tuples
        metric_name: Name of the metric (for header)
        output_path: Where to save the CSV file
    """
    # Get all ensemble sizes across all strategies
    all_sizes = sorted(
        set(size for data in data_by_strategy.values() for size, _ in data)
    )

    # Build lookup dict for quick access: strategy -> {size: value}
    lookup = {}
    for strategy, data in data_by_strategy.items():
        lookup[strategy] = {size: value for size, value in data}

    # Get strategy display labels
    strategies = list(data_by_strategy.keys())
    strategy_labels = [
        STRATEGY_STYLES.get(s, {"label": s})["label"] for s in strategies
    ]

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        header = ["Ensemble Size"] + strategy_labels
        writer.writerow(header)

        # Data rows
        for size in all_sizes:
            row = [size]
            for strategy in strategies:
                value = lookup[strategy].get(size)
                row.append(f"{value:.4f}" if value is not None else "")
            writer.writerow(row)

    typer.echo(f"Table saved to: {output_path}")


@app.command()
def main():
    """Plot ensemble size vs metric for multiple aggregation strategies.

    Configuration is set via constants at the top of this file.

    Example:
        python scripts/figures/plot_ensemble_size_vs_kappa.py
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    evaluate_runs_base = project_root / "artifacts" / "runs" / "evaluate" / RUN_TYPE
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = project_root / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_filename = OUTPUT_FILENAME
    if output_filename is None:
        output_filename = (
            f"ensemble_size_vs_{METRIC_NAME}_multi_strategy.{OUTPUT_FORMAT}"
        )
    output_path = figures_dir / output_filename

    total_runs = sum(len(runs) for runs in RUNS_BY_STRATEGY.values())
    typer.echo(
        f"Analyzing {total_runs} evaluate runs across {len(RUNS_BY_STRATEGY)} strategies"
    )
    typer.echo(f"Base path: {evaluate_runs_base}")
    typer.echo(f"Metric: {METRIC_NAME}\n")

    # Collect data
    data_by_strategy = collect_data_by_strategy(
        RUNS_BY_STRATEGY, evaluate_runs_base, METRIC_NAME
    )

    if not data_by_strategy or all(not data for data in data_by_strategy.values()):
        typer.echo(
            "Error: No data collected. Check that run paths exist and contain valid data.",
            err=True,
        )
        raise typer.Exit(1)

    # Create plot
    plot_ensemble_size_vs_metric_multi_strategy(
        data_by_strategy, METRIC_NAME, output_path, Y_AXIS_LIMITS, PLOT_TITLE
    )

    # Save table
    table_filename = output_filename.replace(f".{OUTPUT_FORMAT}", ".csv")
    table_path = tables_dir / table_filename
    save_data_table(data_by_strategy, METRIC_NAME, table_path)


if __name__ == "__main__":
    app()
