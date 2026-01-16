#!/usr/bin/env python3
"""Plot parallel coordinates of individual model votes.

This script visualizes how 5 LLMs vote on the same examples, showing
consensus patterns and disagreement. Each line represents one example,
connecting the votes across all models.

Dense bands indicate common voting patterns (consensus).
Scattered lines indicate disagreement among models.

Usage:
    python scripts/figures/plot_parallel_coordinates_votes.py
"""

import sys
from pathlib import Path

# Add src to path for imports before other imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import typer

from sqlalchemy.orm import selectinload

from thesis_colors import ENSEMBLE_PALETTE, BHT_COLORS, GRAY_SCALE
from copy_to_overleaf import copy_figure_to_overleaf

# Database access
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import get_session
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    LLMJudgementORM,
)
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    NormalizedDatasetJudgingSampleORM,
)


# ============================================================================
# CONFIGURATION - Edit these constants to change the plot
# ============================================================================

# List of (display_label, run_name) tuples for each model in the ensemble
# Order matters for finding patterns - try placing similar models adjacent
INFER_RUNS = [
    ("Ministral 3B", "ensemble-2-ministral-3b-2515"),
    ("Phi-4", "ensemble-3-phi-4-multimodal-instruct"),
    ("Llama 3.2 3B", "ensemble-4-meta-llama-3.2-3b-instruct-start"),
    ("UI-TARS 7B", "ensemble-5-ui-tars-1.5-7b-start"),
    ("Gemma 3 4B", "ensemble-1-google-gemma-3-4b-it"),
]

# Line transparency (lower = more transparent, helps see density patterns)
# With ~4000 lines, use very low alpha (0.01-0.05)
LINE_ALPHA = 0.03

# Line width
LINE_WIDTH = 0.8

# Color mode: "single" (one color for all lines) or "by_ground_truth" (color by label)
COLOR_MODE = "single"  # "single" or "by_ground_truth"

# Single line color (used when COLOR_MODE = "single")
LINE_COLOR = BHT_COLORS["blue"]

# Colors for ground truth labels (used when COLOR_MODE = "by_ground_truth")
GROUND_TRUTH_COLORS = {
    0: BHT_COLORS["red"],  # Irrelevant
    1: BHT_COLORS["yellow"],  # Relevant
    2: BHT_COLORS["turquoise"],  # Highly Relevant
    3: BHT_COLORS["blue"],  # Perfectly Relevant
}

# Add jitter to y-values to avoid perfect overlap (0.0 = no jitter)
Y_JITTER = 0.08

# Output filename (None = auto-generate)
OUTPUT_FILENAME = "parallel_coordinates_votes.svg"

# Custom plot title (None = auto-generate)
PLOT_TITLE = "Model Voting Patterns Across Ensemble"

# Vote labels for y-axis
VOTE_LABELS = ["0 (Irrelevant)", "1 (Relevant)", "2 (Highly Rel.)", "3 (Perfect)"]

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


def load_ground_truth(run_name: str, session) -> List[int]:
    """Load ground truth labels from an infer run.

    Only needs to be called once since all runs share the same samples.

    Args:
        run_name: Name of the infer run
        session: SQLAlchemy session

    Returns:
        List of ground truth labels (0-3)
    """
    infer_run = session.query(InferRunORM).filter_by(run_name=run_name).first()
    if not infer_run:
        raise ValueError(f"Infer run '{run_name}' not found")

    judgements = (
        session.query(LLMJudgementORM)
        .filter_by(infer_run_output_id=infer_run.infer_run_output_id)
        .order_by(LLMJudgementORM.normalized_dataset_judging_sample_id)
        .all()
    )

    # Batch load all dataset samples in one query
    sample_ids = [j.normalized_dataset_judging_sample_id for j in judgements]
    dataset_samples = (
        session.query(NormalizedDatasetJudgingSampleORM)
        .filter(NormalizedDatasetJudgingSampleORM.id.in_(sample_ids))
        .all()
    )
    sample_by_id = {s.id: s for s in dataset_samples}

    return [
        int(sample_by_id[j.normalized_dataset_judging_sample_id].judging_sample.gold_score)
        for j in judgements
    ]


def collect_all_votes(
    infer_runs: List[Tuple[str, str]],
) -> Tuple[List[str], np.ndarray, List[int]]:
    """Collect votes from all models.

    Args:
        infer_runs: List of (display_label, run_name) tuples

    Returns:
        Tuple of (model_labels, votes_matrix, ground_truth)
        - model_labels: List of display labels
        - votes_matrix: numpy array of shape (n_samples, n_models)
        - ground_truth: List of ground truth labels
    """
    model_labels = []
    all_predictions = []
    ground_truth = None

    engine = get_engine()
    with get_session(engine) as session:
        for label, run_name in infer_runs:
            typer.echo(f"Loading votes from: {run_name}")
            preds = load_votes(run_name, session)

            # Load ground truth only once (from first model)
            if ground_truth is None:
                typer.echo("Loading ground truth...")
                ground_truth = load_ground_truth(run_name, session)

            model_labels.append(label)
            all_predictions.append(preds)

    # Convert to numpy array (n_samples x n_models)
    n_samples = len(ground_truth)
    n_models = len(model_labels)
    votes_matrix = np.zeros((n_samples, n_models), dtype=float)

    for model_idx, preds in enumerate(all_predictions):
        for sample_idx, pred in enumerate(preds):
            if pred is not None:
                votes_matrix[sample_idx, model_idx] = pred
            else:
                votes_matrix[sample_idx, model_idx] = np.nan

    return model_labels, votes_matrix, ground_truth


def plot_parallel_coordinates(
    model_labels: List[str],
    votes_matrix: np.ndarray,
    ground_truth: List[int],
    output_path: Path,
    title: Optional[str] = None,
    alpha: float = 0.03,
    line_width: float = 0.8,
    color_mode: str = "single",
    line_color: str = "#004282",
    gt_colors: Dict[int, str] = None,
    y_jitter: float = 0.0,
):
    """Create parallel coordinates plot of model votes.

    Args:
        model_labels: List of model display names (x-axis)
        votes_matrix: Array of shape (n_samples, n_models) with votes 0-3
        ground_truth: List of ground truth labels (for coloring)
        output_path: Where to save the figure
        title: Optional custom title
        alpha: Line transparency
        line_width: Line width
        color_mode: "single" or "by_ground_truth"
        line_color: Color for single mode
        gt_colors: Dict mapping ground truth to color (for by_ground_truth mode)
        y_jitter: Amount of random jitter to add to y values
    """
    n_samples, n_models = votes_matrix.shape

    fig, ax = plt.subplots(figsize=(12, 7))

    # X positions for each model
    x_positions = np.arange(n_models)

    # Add jitter if requested
    if y_jitter > 0:
        jitter = np.random.uniform(-y_jitter, y_jitter, votes_matrix.shape)
        votes_jittered = votes_matrix + jitter
    else:
        votes_jittered = votes_matrix

    # Plot each sample as a line
    typer.echo(f"Drawing {n_samples:,} lines...")

    for i in range(n_samples):
        y_values = votes_jittered[i, :]

        # Skip if any value is NaN (failed parse)
        if np.any(np.isnan(y_values)):
            continue

        # Determine color
        if color_mode == "by_ground_truth" and gt_colors:
            color = gt_colors.get(ground_truth[i], line_color)
        else:
            color = line_color

        ax.plot(
            x_positions,
            y_values,
            color=color,
            alpha=alpha,
            linewidth=line_width,
            solid_capstyle="round",
        )

    # Configure axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, n_models - 0.5)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(VOTE_LABELS, fontsize=10)
    ax.set_ylim(-0.5, 3.5)

    # Labels
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Vote (Relevance Score)", fontsize=12, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Grid for y-axis only (at vote levels)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add vertical lines at each model position for clarity
    for x in x_positions:
        ax.axvline(x=x, color=GRAY_SCALE["light"], linewidth=1, zorder=0)

    # Add legend if coloring by ground truth
    if color_mode == "by_ground_truth" and gt_colors:
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color=gt_colors[i], linewidth=2, label=VOTE_LABELS[i])
            for i in range(4)
        ]
        ax.legend(
            handles=legend_elements,
            title="Ground Truth",
            loc="upper right",
            fontsize=10,
            framealpha=0.95,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()

    # Copy to Overleaf
    copy_figure_to_overleaf(output_path)


def print_consensus_statistics(votes_matrix: np.ndarray, model_labels: List[str]):
    """Print statistics about voting consensus.

    Args:
        votes_matrix: Array of shape (n_samples, n_models) with votes 0-3
        model_labels: List of model display names
    """
    n_samples, n_models = votes_matrix.shape

    # Filter out samples with NaN
    valid_mask = ~np.any(np.isnan(votes_matrix), axis=1)
    valid_votes = votes_matrix[valid_mask]
    n_valid = len(valid_votes)

    typer.echo("\n" + "=" * 60)
    typer.echo("Consensus Statistics")
    typer.echo("=" * 60)
    typer.echo(f"Total samples: {n_samples:,}")
    typer.echo(f"Valid samples (no parse failures): {n_valid:,}")

    # Count how many models agree on majority vote
    unanimous = 0
    four_agree = 0
    three_agree = 0
    two_agree = 0

    for row in valid_votes:
        votes = row.astype(int)
        unique, counts = np.unique(votes, return_counts=True)
        max_agreement = counts.max()

        if max_agreement == 5:
            unanimous += 1
        elif max_agreement == 4:
            four_agree += 1
        elif max_agreement == 3:
            three_agree += 1
        else:
            two_agree += 1

    typer.echo(f"\nAgreement distribution:")
    typer.echo(f"  5/5 unanimous:  {unanimous:,} ({unanimous/n_valid:.1%})")
    typer.echo(f"  4/5 agree:      {four_agree:,} ({four_agree/n_valid:.1%})")
    typer.echo(f"  3/5 agree:      {three_agree:,} ({three_agree/n_valid:.1%})")
    typer.echo(f"  2/5 or less:    {two_agree:,} ({two_agree/n_valid:.1%})")

    # Pairwise agreement
    typer.echo(f"\nPairwise agreement rates:")
    for i in range(n_models):
        for j in range(i + 1, n_models):
            agree = np.sum(valid_votes[:, i] == valid_votes[:, j])
            rate = agree / n_valid
            typer.echo(f"  {model_labels[i]} vs {model_labels[j]}: {rate:.1%}")


@app.command()
def main():
    """Plot parallel coordinates of model votes.

    Configuration is set via constants at the top of this file.
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_filename = OUTPUT_FILENAME
    if output_filename is None:
        output_filename = "parallel_coordinates_votes.svg"
    output_path = figures_dir / output_filename

    typer.echo(f"Loading votes from {len(INFER_RUNS)} models...")

    # Collect all votes
    model_labels, votes_matrix, ground_truth = collect_all_votes(INFER_RUNS)

    typer.echo(
        f"Collected {votes_matrix.shape[0]:,} samples from {len(model_labels)} models"
    )

    # Print statistics
    print_consensus_statistics(votes_matrix, model_labels)

    # Create plot
    plot_parallel_coordinates(
        model_labels=model_labels,
        votes_matrix=votes_matrix,
        ground_truth=ground_truth,
        output_path=output_path,
        title=PLOT_TITLE,
        alpha=LINE_ALPHA,
        line_width=LINE_WIDTH,
        color_mode=COLOR_MODE,
        line_color=LINE_COLOR,
        gt_colors=GROUND_TRUTH_COLORS,
        y_jitter=Y_JITTER,
    )


if __name__ == "__main__":
    app()
