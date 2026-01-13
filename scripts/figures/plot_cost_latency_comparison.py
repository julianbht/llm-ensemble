#!/usr/bin/env python3
"""
Generate stacked bar chart comparing cost and latency across models and one ensemble.

Reads extracted cost/latency data and produces a 2-panel figure showing:
- Left panel: Total cost (USD) comparison
- Right panel: Total latency (seconds) comparison

Single models show as solid bars. The ensemble shows as a stacked bar with member contributions.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

from thesis_colors import BHT_COLORS, GRAY_SCALE, ENSEMBLE_PALETTE, REFERENCE_COLOR
from copy_to_overleaf import copy_figure_to_overleaf

# Configuration
INPUT_FILE = Path("artifacts/other/cost_latency_comparison.json")
OUTPUT_FILE = Path("artifacts/figures/cost_latency_comparison.svg")

# Which runs to show on x-axis (in order)
RUN_ORDER = [
    "reference-ensemble-gpt-5-1-all-samples-start",  # Reference model
    "ensemble",  # The ensemble (will be stacked from member runs)
]

# Ensemble member runs (these will be stacked together)
ENSEMBLE_MEMBER_RUNS = [
    "ensemble-1-google-gemma-3-4b-it",
    "ensemble-2-ministral-3b-2515",
    "ensemble-3-phi-4-multimodal-instruct",
    "ensemble-4-meta-llama-3.2-3b-instruct-start",
    "ensemble-5-ui-tars-1.5-7b-start",
]

# Color scheme (using BHT thesis colors)
MODEL_COLORS = {
    "ensemble-1-google-gemma-3-4b-it": ENSEMBLE_PALETTE[0],
    "ensemble-2-ministral-3b-2515": ENSEMBLE_PALETTE[1],
    "ensemble-3-phi-4-multimodal-instruct": ENSEMBLE_PALETTE[2],
    "ensemble-4-meta-llama-3.2-3b-instruct-start": ENSEMBLE_PALETTE[3],
    "ensemble-5-ui-tars-1.5-7b-start": ENSEMBLE_PALETTE[4],
    "reference-ensemble-gpt-5-1-all-samples-start": REFERENCE_COLOR,
}

# Display labels for x-axis and legend (using model_id without provider prefix)
DISPLAY_LABELS = {
    "reference-ensemble-gpt-5-1-all-samples-start": "GPT-5.1",
    "ensemble": "Ensemble",
    "ensemble-1-google-gemma-3-4b-it": "gemma-3-4b-it",
    "ensemble-2-ministral-3b-2515": "ministral-3b-2512",
    "ensemble-3-phi-4-multimodal-instruct": "phi-4-multimodal-instruct",
    "ensemble-4-meta-llama-3.2-3b-instruct-start": "llama-3.2-3b-instruct",
    "ensemble-5-ui-tars-1.5-7b-start": "ui-tars-1.5-7b",
}

# Figure dimensions
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 5
DPI = 300


def load_data(input_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load extracted cost/latency data from JSON.

    Args:
        input_path: Path to JSON file

    Returns:
        Dict mapping run_name to metrics
    """
    with input_path.open("r") as f:
        metrics_list = json.load(f)

    # Convert list to dict keyed by run_name
    return {m["run_name"]: m for m in metrics_list}


def create_comparison_chart(
    metrics: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """Create 2-panel comparison chart.

    Args:
        metrics: Dict mapping run_name to metrics
        output_path: Where to save the figure
    """
    fig, (ax_cost, ax_latency) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    x_pos = np.arange(len(RUN_ORDER))
    bar_width = 0.6

    # Prepare data for plotting
    cost_values = []
    latency_values = []
    labels = []
    legend_mapping = {}  # Track which model gets which color (for legend)

    for run_name in RUN_ORDER:
        labels.append(DISPLAY_LABELS.get(run_name, run_name))

        if run_name == "ensemble":
            # Stack ensemble members
            total_cost = sum(
                metrics[m]["total_cost_usd"]
                for m in ENSEMBLE_MEMBER_RUNS
                if m in metrics
            )
            total_latency = sum(
                metrics[m]["total_latency_ms"]
                for m in ENSEMBLE_MEMBER_RUNS
                if m in metrics
            ) / (
                1000.0 * 3600.0
            )  # Convert to hours

            cost_values.append(("ensemble", total_cost, ENSEMBLE_MEMBER_RUNS))
            latency_values.append(("ensemble", total_latency, ENSEMBLE_MEMBER_RUNS))
        else:
            # Single model
            cost_values.append(
                ("single", metrics[run_name]["total_cost_usd"], run_name)
            )
            latency_values.append(
                (
                    "single",
                    metrics[run_name]["total_latency_ms"] / (1000.0 * 3600.0),
                    run_name,
                )
            )  # Convert to hours

    # Left panel: Cost
    for i, (bar_type, value, info) in enumerate(cost_values):
        if bar_type == "single":
            # Solid bar for single model
            color = MODEL_COLORS.get(info, GRAY_SCALE["very_light"])
            ax_cost.bar(
                x_pos[i],
                value,
                bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
        else:
            # Stacked bar for ensemble - sort by cost (largest at bottom)
            member_costs = [
                (m, metrics[m]["total_cost_usd"]) for m in info if m in metrics
            ]
            member_costs.sort(key=lambda x: x[1], reverse=True)  # Largest first

            bottom = 0
            for idx, (member_run, member_cost) in enumerate(member_costs):
                # Assign colors based on sorted position (largest gets first color)
                color = (
                    ENSEMBLE_PALETTE[idx]
                    if idx < len(ENSEMBLE_PALETTE)
                    else GRAY_SCALE["very_light"]
                )
                legend_mapping[member_run] = color  # Track for legend
                ax_cost.bar(
                    x_pos[i],
                    member_cost,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottom += member_cost

    ax_cost.set_ylabel("Total Cost (USD)", fontsize=11, fontweight="bold")
    ax_cost.set_xlabel("Model / Ensemble", fontsize=11, fontweight="bold")
    ax_cost.set_xticks(x_pos)
    ax_cost.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax_cost.grid(axis="y", alpha=0.3, linestyle="--")
    ax_cost.set_axisbelow(True)

    # Right panel: Latency
    for i, (bar_type, value, info) in enumerate(latency_values):
        if bar_type == "single":
            # Solid bar for single model
            color = MODEL_COLORS.get(info, GRAY_SCALE["very_light"])
            ax_latency.bar(
                x_pos[i],
                value,
                bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
        else:
            # Stacked bar for ensemble - sort by latency (largest at bottom)
            member_latencies = [
                (m, metrics[m]["total_latency_ms"] / (1000.0 * 3600.0))
                for m in info
                if m in metrics
            ]  # Convert to hours
            member_latencies.sort(key=lambda x: x[1], reverse=True)  # Largest first

            bottom = 0
            for idx, (member_run, member_latency) in enumerate(member_latencies):
                # Assign colors based on sorted position (largest gets first color)
                color = (
                    ENSEMBLE_PALETTE[idx]
                    if idx < len(ENSEMBLE_PALETTE)
                    else GRAY_SCALE["very_light"]
                )
                ax_latency.bar(
                    x_pos[i],
                    member_latency,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottom += member_latency

    ax_latency.set_ylabel("Total Latency (hours)", fontsize=11, fontweight="bold")
    ax_latency.set_xlabel("Model / Ensemble", fontsize=11, fontweight="bold")
    ax_latency.set_xticks(x_pos)
    ax_latency.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax_latency.grid(axis="y", alpha=0.3, linestyle="--")
    ax_latency.set_axisbelow(True)

    # Add legend using the tracked color mapping
    legend_handles = []
    legend_labels = []
    for run in ENSEMBLE_MEMBER_RUNS:
        if run in legend_mapping:
            legend_handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc=legend_mapping[run],
                    edgecolor="black",
                    linewidth=0.5,
                )
            )
            legend_labels.append(DISPLAY_LABELS.get(run, run))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.close()


def main():
    """Main execution."""
    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_path = project_root / INPUT_FILE
    output_path = project_root / OUTPUT_FILE

    print(f"Reading data from: {input_path}")

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run extract_cost_latency_data.py first to generate the data.")
        return

    # Load data
    metrics = load_data(input_path)
    print(f"Loaded metrics for {len(metrics)} runs")

    # Verify all required runs are present (skip "ensemble" which is synthetic)
    missing_runs = [
        run for run in RUN_ORDER if run != "ensemble" and run not in metrics
    ]
    if missing_runs:
        print(f"Warning: Missing data for runs: {missing_runs}")

    # Verify all ensemble member runs are present
    missing_members = [run for run in ENSEMBLE_MEMBER_RUNS if run not in metrics]
    if missing_members:
        print(f"Warning: Missing ensemble member runs: {missing_members}")

    # Create plot
    print("Generating figure...")
    create_comparison_chart(metrics, output_path)

    # Copy to Overleaf
    copy_figure_to_overleaf(output_path, project_root)

    print("\nDone!")


if __name__ == "__main__":
    main()
