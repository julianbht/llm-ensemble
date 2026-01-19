#!/usr/bin/env python3
"""Plot grouped bar chart comparing repeat runs of the same model to determine ground noise.

This script reads evaluate_run.json files from multiple repeat runs of the same models
and creates a grouped bar chart where each model's runs are displayed side-by-side,
making it immediately obvious how much the kappa values differ between runs.

Usage:
    # Run with defaults from constants below
    python scripts/figures/plot_noise_agreement_comparison.py

    # Override metric
    python scripts/figures/plot_noise_agreement_comparison.py --metric krippendorffs_alpha
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import typer

from thesis_colors import BHT_COLORS, GRAY_SCALE, ENSEMBLE_PALETTE
from copy_to_overleaf import copy_figure_to_overleaf


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================


@dataclass
class ModelRunGroup:
    """A group of repeat runs for the same model."""

    model_label: str  # Display label for the model (e.g., "phi-4-multimodal")
    run_names: List[str]  # List of run names to compare


# Define groups of runs to compare
# Each group contains repeat runs of the same model
RUN_GROUPS = [
    ModelRunGroup(
        model_label="gemma-3-4b",
        run_names=[
            "ensemble-1-google-gemma-3-4b-it",
            "noise-1-ensemble-1-google-gemma-3-4b-it-start",
            "+2-noise-1-google-gemma-3-4b-it-start",
        ],
    ),
    ModelRunGroup(
        model_label="ministral-3b",
        run_names=[
            "ensemble-2-ministral-3b-2515",
            "noise-2-ensemble-2-ministral-3b-2512-start",
            "+2-noise-2-ministral-3b-2512",
        ],
    ),
    ModelRunGroup(
        model_label="phi-4-multimodal",
        run_names=[
            "ensemble-3-phi-4-multimodal-instruct",
            "noise-3-ensemble-3-phi-4-multimodal-instruct-start",
            # Third run had API issues - no data available
        ],
    ),
    ModelRunGroup(
        model_label="llama-3.2-3b",
        run_names=[
            "ensemble-4-meta-llama-3.2-3b-instruct-start",
            "noise-4-ensemble-4-meta-llama-3.2-3b-instruct-start",
            "+2-noise-4-meta-llama-3.2-3b-instruct-start",
        ],
    ),
    ModelRunGroup(
        model_label="ui-tars-1.5-7b",
        run_names=[
            "ensemble-5-ui-tars-1.5-7b-start",
            "noise-5-ensemble-5-ui-tars-1.5-7b-start",
            "+2-noise-5-ui-tars-1.5-7b-start",
        ],
    ),
    ModelRunGroup(
        model_label="GPT-5.1",
        run_names=[
            "reference-ensemble-gpt-5-1-all-samples-start",
            "noise-reference-ensemble-gpt-5-1-all-samples-start",
            "+2-noise-reference-ensemble-gpt-5-1-all-samples-start",
        ],
    ),
]

# Metric to plot
METRIC_NAME = "cohens_kappa"

# Run type directory
RUN_TYPE = "official"

# Y-axis limits: None for auto-scaling, or [min, max] to fix the range
Y_AXIS_LIMITS = [0.0, 0.5]

# Output filename (None = auto-generate)
OUTPUT_FILENAME = None

# Custom plot title (None = auto-generate)
PLOT_TITLE = "Run-to-Run Agreement Variability"

# LaTeX caption for the figure
LATEX_CAPTION = r"\caption{\CK for each model across repeated runs, computed against human annotations.}"

# Bar appearance
BAR_WIDTH = 0.35  # Width of individual bars
GROUP_SPACING = 0.15  # Extra space between groups

# ============================================================================


app = typer.Typer()


def read_evaluate_run(run_path: Path) -> dict:
    """Read and parse evaluate_run.json file."""
    with open(run_path / "evaluate_run.json", "r") as f:
        return json.load(f)


def extract_metric_value(evaluate_run: dict, metric_name: str) -> float:
    """Extract specified metric value from metric results."""
    for metric in evaluate_run["metric_results"]:
        if metric["name"] == metric_name:
            return metric["value"]
    raise ValueError(f"Metric '{metric_name}' not found in metric results")


@dataclass
class GroupData:
    """Collected data for a model group."""

    model_label: str
    run_labels: List[str]  # Short labels for each run (e.g., "Run 1", "Run 2")
    values: List[float]


def collect_group_data(
    run_groups: List[ModelRunGroup],
    evaluate_runs_base: Path,
    metric_name: str,
) -> List[GroupData]:
    """Collect metric values for all run groups.

    Args:
        run_groups: List of model run groups to process
        evaluate_runs_base: Base path for evaluate runs
        metric_name: Name of metric to extract

    Returns:
        List of GroupData with collected values
    """
    collected = []

    for group in run_groups:
        run_labels = []
        values = []

        for i, run_name in enumerate(group.run_names, start=1):
            run_path = evaluate_runs_base / run_name
            if not run_path.exists():
                typer.echo(f"Warning: Run path does not exist: {run_path}", err=True)
                continue

            try:
                evaluate_run = read_evaluate_run(run_path)
                metric_value = extract_metric_value(evaluate_run, metric_name)
                run_labels.append(f"Run {i}")
                values.append(metric_value)
                typer.echo(
                    f"Found: {group.model_label} Run {i} = {metric_value:.4f} ({run_name})"
                )
            except (ValueError, FileNotFoundError) as e:
                typer.echo(f"Warning: {e} for run {run_name}", err=True)
                continue

        if values:
            collected.append(
                GroupData(
                    model_label=group.model_label,
                    run_labels=run_labels,
                    values=values,
                )
            )

    return collected


def get_run_colors(num_runs: int) -> List[str]:
    """Get colors for runs within a group: blue, yellow, turquoise."""
    colors = [BHT_COLORS["blue"], BHT_COLORS["yellow"], BHT_COLORS["turquoise"]]
    return colors[:num_runs]


def plot_grouped_bar_chart(
    group_data: List[GroupData],
    metric_name: str,
    output_path: Path,
    y_limits: Optional[List[float]] = None,
    title: Optional[str] = None,
    bar_width: float = 0.35,
    group_spacing: float = 0.15,
):
    """Create and save grouped bar chart for noise comparison.

    Args:
        group_data: List of GroupData with model groups and their values
        metric_name: Name of the metric (for axis label)
        output_path: Where to save the figure
        y_limits: Y-axis limits [min, max] or None for auto-scaling
        title: Optional custom title
        bar_width: Width of individual bars
        group_spacing: Extra space between groups
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Find max runs per group for positioning
    max_runs = max(len(g.values) for g in group_data)

    # Calculate x positions for groups
    # Each group needs space for max_runs bars plus group spacing
    group_width = max_runs * bar_width + group_spacing
    x_group_centers = np.arange(len(group_data)) * group_width

    # Plot bars for each group
    all_bars = []
    for group_idx, group in enumerate(group_data):
        colors = get_run_colors(len(group.values))
        group_center = x_group_centers[group_idx]

        # Calculate bar positions within group (centered)
        num_bars = len(group.values)
        total_bar_width = num_bars * bar_width
        start_x = group_center - total_bar_width / 2 + bar_width / 2

        for bar_idx, (value, color) in enumerate(zip(group.values, colors)):
            x_pos = start_x + bar_idx * bar_width
            bar = ax.bar(
                x_pos,
                value,
                bar_width * 0.9,  # Slight gap between bars
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            all_bars.append((bar, value))

            # Add value label on bar
            ax.text(
                x_pos,
                value + 0.005,  # Small offset above bar
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                rotation=0,
            )

    # Calculate and display range for each group
    for group_idx, group in enumerate(group_data):
        if len(group.values) >= 2:
            val_range = max(group.values) - min(group.values)
            group_center = x_group_centers[group_idx]
            max_val = max(group.values)

            # Add range annotation
            ax.annotate(
                f"range={val_range:.3f}",
                xy=(group_center, max_val + 0.035),
                ha="center",
                va="bottom",
                fontsize=9,
                color=BHT_COLORS["red"] if val_range > 0.02 else GRAY_SCALE["dark"],
                fontweight="bold",
            )

    # Set y-axis limits
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    # Labels and title
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    metric_display = metric_name.replace("_", " ").title()
    ax.set_ylabel(f"{metric_display}", fontsize=12, fontweight="bold")

    if title is None:
        title = f"Run-to-Run Variability: {metric_display}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Set x-tick labels to model names
    ax.set_xticks(x_group_centers)
    ax.set_xticklabels([g.model_label for g in group_data], rotation=45, ha="right")

    # Grid
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    legend_patches = [
        mpatches.Patch(color=BHT_COLORS["blue"], label="Run 1"),
        mpatches.Patch(color=BHT_COLORS["yellow"], label="Run 2"),
    ]
    if max_runs > 2:
        legend_patches.append(
            mpatches.Patch(color=BHT_COLORS["turquoise"], label="Run 3")
        )
    ax.legend(handles=legend_patches, loc="upper right", fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()

    # Copy to Overleaf
    copy_figure_to_overleaf(output_path)


def print_summary_statistics(group_data: List[GroupData], metric_name: str):
    """Print summary statistics about run-to-run variability."""
    typer.echo("\n" + "=" * 60)
    typer.echo("SUMMARY: Run-to-Run Variability")
    typer.echo("=" * 60)

    all_ranges = []
    for group in group_data:
        if len(group.values) >= 2:
            # Print all run values
            run_strs = [f"Run{i+1}={v:.4f}" for i, v in enumerate(group.values)]
            val_range = max(group.values) - min(group.values)
            all_ranges.append(val_range)
            typer.echo(
                f"{group.model_label:20s}: "
                f"{', '.join(run_strs)}, "
                f"range={val_range:.4f}"
            )

    if all_ranges:
        typer.echo("-" * 60)
        typer.echo(f"Mean range across runs: {np.mean(all_ranges):.4f}")
        typer.echo(f"Max range:              {np.max(all_ranges):.4f}")
        typer.echo(f"Min range:              {np.min(all_ranges):.4f}")


@app.command()
def main(
    metric: str = typer.Option(METRIC_NAME, "--metric", "-m", help="Metric to plot"),
):
    """Plot grouped bar chart comparing repeat runs of the same model.

    This helps determine ground noise by showing how much the agreement metric
    varies between identical runs of the same model.
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    evaluate_runs_base = project_root / "artifacts" / "runs" / "evaluate" / RUN_TYPE
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_filename = OUTPUT_FILENAME
    if output_filename is None:
        output_filename = f"noise_agreement_comparison_{metric}.svg"
    output_path = figures_dir / output_filename

    typer.echo(f"Analyzing {len(RUN_GROUPS)} model groups from: {evaluate_runs_base}")
    typer.echo(f"Metric: {metric}\n")

    # Collect data
    group_data = collect_group_data(RUN_GROUPS, evaluate_runs_base, metric)

    if not group_data:
        typer.echo(
            "Error: No data collected. Check that run paths exist and contain valid data.",
            err=True,
        )
        typer.echo(
            "\nHint: Make sure the noise runs have been evaluated (run evaluate CLI).",
            err=True,
        )
        raise typer.Exit(1)

    # Print summary statistics
    print_summary_statistics(group_data, metric)

    # Create plot
    plot_grouped_bar_chart(
        group_data,
        metric,
        output_path,
        Y_AXIS_LIMITS,
        PLOT_TITLE,
        BAR_WIDTH,
        GROUP_SPACING,
    )


if __name__ == "__main__":
    app()
