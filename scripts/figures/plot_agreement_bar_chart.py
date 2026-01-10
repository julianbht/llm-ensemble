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
    ("Ensemble Size 2", "2-ensemble-first-figure-check"),
    ("Ensemble Size 3", "3-ensemble-first-figure-check"),
]

# Metric to plot (must exist in evaluate_run.json metric_results)
METRIC_NAME = "cohens_kappa"

# Run type directory (subdirectory under artifacts/runs/evaluate/)
RUN_TYPE = "test"

# Output filename (None = auto-generate from metric name)
OUTPUT_FILENAME = None  # e.g., "my_figure.svg" or None

# Custom plot title (None = auto-generate)
PLOT_TITLE = None  # e.g., "Individual vs Ensemble Agreement"

# ============================================================================


app = typer.Typer()


def read_evaluate_run(run_path: Path) -> dict:
    """Read and parse evaluate_run.json file."""
    with open(run_path / "evaluate_run.json", 'r') as f:
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


def get_metric_metadata(evaluate_run: dict, metric_name: str) -> dict:
    """Extract metric metadata for visualization.

    Returns dict with min_value, max_value, higher_is_better if available.
    """
    for metric in evaluate_run["metric_results"]:
        if metric["name"] == metric_name:
            return {
                "min_value": metric.get("min_value"),
                "max_value": metric.get("max_value"),
                "higher_is_better": metric.get("higher_is_better", True),
                "description": metric.get("description", metric_name)
            }
    return {"min_value": None, "max_value": None, "higher_is_better": True, "description": metric_name}


def collect_data(
    runs: List[Tuple[str, str]],
    evaluate_runs_base: Path,
    metric_name: str
) -> Tuple[List[str], List[float], dict]:
    """Collect (label, metric_value) pairs from evaluate runs.

    Args:
        runs: List of (label, run_name) tuples
        evaluate_runs_base: Base path for evaluate runs
        metric_name: Name of metric to extract

    Returns:
        Tuple of (labels, values, metadata)
    """
    labels = []
    values = []
    metadata = None

    for label, run_name in runs:
        run_path = evaluate_runs_base / run_name
        if not run_path.exists():
            typer.echo(f"Warning: Run path does not exist: {run_path}", err=True)
            continue

        evaluate_run = read_evaluate_run(run_path)

        # Extract metadata from first run
        if metadata is None:
            metadata = get_metric_metadata(evaluate_run, metric_name)

        try:
            metric_value = extract_metric_value(evaluate_run, metric_name)
            labels.append(label)
            values.append(metric_value)
            typer.echo(f"Found: {label} = {metric_value:.4f} ({run_name})")
        except ValueError as e:
            typer.echo(f"Warning: {e} for run {run_name}", err=True)
            continue

    return labels, values, metadata


def plot_agreement_bar_chart(
    labels: List[str],
    values: List[float],
    metadata: dict,
    metric_name: str,
    output_path: Path,
    title: Optional[str] = None,
):
    """Create and save bar chart of agreement metrics.

    Args:
        labels: Model/run labels for x-axis
        values: Metric values for each model
        metadata: Metric metadata (min_value, max_value, higher_is_better)
        metric_name: Name of the metric (for axis label)
        output_path: Where to save the figure
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(labels, values, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Highlight the last bar (assumed to be ensemble) in different color
    if len(bars) > 1:
        bars[-1].set_color('coral')
        bars[-1].set_alpha(0.9)

    # Set y-axis limits from metadata if available
    y_min = metadata.get("min_value")
    y_max = metadata.get("max_value")
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    # Labels and title
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')

    metric_display = metric_name.replace('_', ' ').title()
    direction = "(higher is better)" if metadata.get("higher_is_better", True) else "(lower is better)"
    ax.set_ylabel(f'{metric_display} {direction}', fontsize=12, fontweight='bold')

    if title is None:
        title = f'Agreement Metrics: {metric_display}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Grid for readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Rotate x-labels if many models
    if len(labels) > 4:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()


@app.command()
def main(
    metric: Optional[str] = typer.Option(
        None,
        help="Metric name to plot (overrides METRIC_NAME constant)"
    ),
    run_type: Optional[str] = typer.Option(
        None,
        help="Run type directory (overrides RUN_TYPE constant)"
    ),
    output: Optional[str] = typer.Option(
        None,
        help="Output filename (overrides OUTPUT_FILENAME constant)"
    ),
    title: Optional[str] = typer.Option(
        None,
        help="Custom plot title (overrides PLOT_TITLE constant)"
    ),
):
    """Plot agreement metric bar chart comparing models.

    Configuration is set via constants at the top of this file.
    CLI options can override those constants if needed.

    Example:
        # Use configuration from constants
        python scripts/figures/plot_agreement_bar_chart.py

        # Override metric only
        python scripts/figures/plot_agreement_bar_chart.py --metric krippendorffs_alpha
    """
    # Use constants as defaults, allow CLI overrides
    metric_name = metric if metric is not None else METRIC_NAME
    run_type_dir = run_type if run_type is not None else RUN_TYPE
    output_filename = output if output is not None else OUTPUT_FILENAME
    plot_title = title if title is not None else PLOT_TITLE

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    evaluate_runs_base = project_root / "artifacts" / "runs" / "evaluate" / run_type_dir
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Default output filename if not specified
    if output_filename is None:
        output_filename = f"agreement_bar_chart_{metric_name}.svg"
    output_path = figures_dir / output_filename

    # Use RUNS constant (no CLI override for simplicity)
    parsed_runs = RUNS

    typer.echo(f"Analyzing {len(parsed_runs)} evaluate runs from: {evaluate_runs_base}")
    typer.echo(f"Metric: {metric_name}\n")

    # Collect data
    labels, values, metadata = collect_data(parsed_runs, evaluate_runs_base, metric_name)

    if not labels:
        typer.echo("Error: No data collected. Check that run paths exist and contain valid data.", err=True)
        raise typer.Exit(1)

    # Create plot
    plot_agreement_bar_chart(labels, values, metadata, metric_name, output_path, plot_title)


if __name__ == "__main__":
    app()
