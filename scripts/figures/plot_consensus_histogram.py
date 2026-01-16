#!/usr/bin/env python3
"""Plot histogram of consensus strength across ensemble models.

This script shows how often models agree on their votes. For each example,
it counts how many models voted for the most common (plurality) vote.

Interpretation:
- Spike at 5/5 → strong overall consensus
- Large mass at 3/5 → frequent ambiguity
- Mass at 2/5 or below → high disagreement

Usage:
    python scripts/figures/plot_consensus_histogram.py
"""

import sys
from pathlib import Path

# Add src to path for imports before other imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import typer

from sqlalchemy.orm import selectinload

from thesis_colors import BHT_COLORS, GRAY_SCALE
from copy_to_overleaf import copy_figure_to_overleaf

# Database access
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import get_session
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    LLMJudgementORM,
)


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================

# List of (display_label, run_name) tuples for each model in the ensemble
INFER_RUNS = [
    ("Gemma 3 4B", "ensemble-1-google-gemma-3-4b-it"),
    ("Ministral 3B", "ensemble-2-ministral-3b-2515"),
    ("Phi-4", "ensemble-3-phi-4-multimodal-instruct"),
    ("Llama 3.2 3B", "ensemble-4-meta-llama-3.2-3b-instruct-start"),
    ("UI-TARS 7B", "ensemble-5-ui-tars-1.5-7b-start"),
]

# Bar color
BAR_COLOR = BHT_COLORS["blue"]

# Output filename (None = auto-generate)
OUTPUT_FILENAME = "consensus_histogram.svg"

# Custom plot title (None = auto-generate)
PLOT_TITLE = "Consensus Strength Distribution"

# Y-axis limits: None for auto-scaling, or [min, max]
Y_AXIS_LIMITS = None

# Show percentage labels on bars
SHOW_PERCENTAGES = True

# ============================================================================


app = typer.Typer()


def load_votes(run_name: str, session) -> List[Optional[int]]:
    """Load just the LLM votes from an infer run.

    Args:
        run_name: Name of the infer run
        session: SQLAlchemy session

    Returns:
        List of votes (0-3) or None for parse failures
    """
    infer_run = session.query(InferRunORM).filter_by(run_name=run_name).first()
    if not infer_run:
        raise ValueError(f"Infer run '{run_name}' not found")

    judgements = (
        session.query(LLMJudgementORM)
        .filter_by(infer_run_output_id=infer_run.infer_run_output_id)
        .options(selectinload(LLMJudgementORM.llm_score))
        .order_by(LLMJudgementORM.normalized_dataset_judging_sample_id)
        .all()
    )

    return [
        int(j.llm_score.label) if j.llm_score and j.llm_score.label is not None else None
        for j in judgements
    ]


def collect_all_votes(
    infer_runs: List[Tuple[str, str]]
) -> Tuple[List[str], np.ndarray]:
    """Collect votes from all models.

    Args:
        infer_runs: List of (display_label, run_name) tuples

    Returns:
        Tuple of (model_labels, votes_matrix)
        - model_labels: List of display labels
        - votes_matrix: numpy array of shape (n_samples, n_models)
    """
    model_labels = []
    all_predictions = []

    engine = get_engine()
    with get_session(engine) as session:
        for label, run_name in infer_runs:
            typer.echo(f"Loading votes from: {run_name}")
            preds = load_votes(run_name, session)
            model_labels.append(label)
            all_predictions.append(preds)

    # Convert to numpy array (n_samples x n_models)
    n_samples = len(all_predictions[0])
    n_models = len(model_labels)
    votes_matrix = np.zeros((n_samples, n_models), dtype=float)

    for model_idx, preds in enumerate(all_predictions):
        for sample_idx, pred in enumerate(preds):
            if pred is not None:
                votes_matrix[sample_idx, model_idx] = pred
            else:
                votes_matrix[sample_idx, model_idx] = np.nan

    return model_labels, votes_matrix


def compute_consensus_strength(votes_matrix: np.ndarray) -> List[int]:
    """Compute consensus strength for each sample.

    Args:
        votes_matrix: Array of shape (n_samples, n_models)

    Returns:
        List of max agreement counts (1-5) for each valid sample
    """
    n_samples, n_models = votes_matrix.shape
    consensus_counts = []

    for i in range(n_samples):
        row = votes_matrix[i, :]

        # Skip if any NaN
        if np.any(np.isnan(row)):
            continue

        votes = row.astype(int)
        _, counts = np.unique(votes, return_counts=True)
        max_agreement = counts.max()
        consensus_counts.append(max_agreement)

    return consensus_counts


def plot_consensus_histogram(
    consensus_counts: List[int],
    n_models: int,
    output_path: Path,
    title: Optional[str] = None,
    bar_color: str = "#004282",
    y_limits: Optional[List[float]] = None,
    show_percentages: bool = True,
):
    """Create histogram of consensus strength.

    Args:
        consensus_counts: List of max agreement counts (1 to n_models)
        n_models: Number of models in ensemble
        output_path: Where to save the figure
        title: Optional custom title
        bar_color: Color for bars
        y_limits: Optional y-axis limits
        show_percentages: Whether to show percentage labels on bars
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count occurrences of each consensus level
    counter = Counter(consensus_counts)
    total = len(consensus_counts)

    # X positions and heights
    x_labels = [f"{i}/{n_models}" for i in range(1, n_models + 1)]
    x_positions = list(range(1, n_models + 1))
    heights = [counter.get(i, 0) for i in range(1, n_models + 1)]

    # Create bars
    bars = ax.bar(
        x_positions,
        heights,
        color=bar_color,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar, height in zip(bars, heights):
        if height > 0:
            if show_percentages:
                pct = height / total * 100
                label = f"{height:,}\n({pct:.1f}%)"
            else:
                label = f"{height:,}"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # Set y-axis limits
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    # Labels and title
    ax.set_xlabel("Consensus Strength (models agreeing on plurality vote)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Examples", fontsize=12, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()

    # Copy to Overleaf
    copy_figure_to_overleaf(output_path)


def print_consensus_summary(consensus_counts: List[int], n_models: int):
    """Print summary statistics."""
    counter = Counter(consensus_counts)
    total = len(consensus_counts)

    typer.echo("\n" + "=" * 50)
    typer.echo("Consensus Strength Summary")
    typer.echo("=" * 50)
    typer.echo(f"Total valid samples: {total:,}")
    typer.echo("")

    for i in range(n_models, 0, -1):
        count = counter.get(i, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        typer.echo(f"  {i}/{n_models} agree: {count:>5,} ({pct:>5.1f}%) {bar}")

    # Summary interpretation
    unanimous = counter.get(n_models, 0)
    majority = sum(counter.get(i, 0) for i in range(3, n_models + 1))
    low_agreement = sum(counter.get(i, 0) for i in range(1, 3))

    typer.echo("")
    typer.echo(f"Unanimous (5/5):     {unanimous:,} ({unanimous/total:.1%})")
    typer.echo(f"Majority (3-5/5):    {majority:,} ({majority/total:.1%})")
    typer.echo(f"Low agreement (1-2): {low_agreement:,} ({low_agreement/total:.1%})")


@app.command()
def main():
    """Plot consensus strength histogram.

    Configuration is set via constants at the top of this file.
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_filename = OUTPUT_FILENAME or "consensus_histogram.svg"
    output_path = figures_dir / output_filename

    typer.echo(f"Loading votes from {len(INFER_RUNS)} models...")

    # Collect all votes
    model_labels, votes_matrix = collect_all_votes(INFER_RUNS)
    n_models = len(model_labels)

    typer.echo(f"Collected {votes_matrix.shape[0]:,} samples from {n_models} models")

    # Compute consensus strength
    consensus_counts = compute_consensus_strength(votes_matrix)

    # Print summary
    print_consensus_summary(consensus_counts, n_models)

    # Create plot
    plot_consensus_histogram(
        consensus_counts=consensus_counts,
        n_models=n_models,
        output_path=output_path,
        title=PLOT_TITLE,
        bar_color=BAR_COLOR,
        y_limits=Y_AXIS_LIMITS,
        show_percentages=SHOW_PERCENTAGES,
    )


if __name__ == "__main__":
    app()
