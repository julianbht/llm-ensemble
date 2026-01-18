#!/usr/bin/env python3
"""Plot pairwise Cohen's kappa for self-agreement across repeated runs.

Computes Cohen's kappa between each pair of runs (Run1↔Run2, Run1↔Run3, Run2↔Run3)
to show how consistently models judge the same items. Visualizes with:
- Dumbbell chart: shows range (min to max) with all points
- Dot plot with jitter: shows all pairwise kappa values

Usage:
    python scripts/figures/plot_self_agreement_cohens_kappa.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

import numpy as np
import matplotlib.pyplot as plt
import typer
from sklearn.metrics import cohen_kappa_score
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from thesis_colors import BHT_COLORS, GRAY_SCALE
from copy_to_overleaf import copy_figure_to_overleaf

from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    InferRunOutputORM,
    LLMJudgementORM,
    LLMScoreORM,
)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ModelRunGroup:
    """A group of repeat runs for the same model."""

    model_label: str
    run_names: list[str]


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
        model_label="phi-4",
        run_names=[
            "ensemble-3-phi-4-multimodal-instruct",
            "noise-3-ensemble-3-phi-4-multimodal-instruct-start",
            "+2-noise-3-phi-4-multimodal-instruct-start",
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

OUTPUT_DUMBBELL = "self_agreement_kappa_dumbbell.svg"
OUTPUT_DOTPLOT = "self_agreement_kappa_dotplot.svg"
PLOT_TITLE = "Model Self-Agreement (Pairwise Cohen's κ)"


# ============================================================================
# DATA LOADING
# ============================================================================


@dataclass
class RunData:
    """Data loaded from a single infer run."""

    run_name: str
    sample_fingerprint: str
    sample_ids: list[str]
    predictions: list[int | None]


def load_run_data(run_name: str, engine) -> RunData:
    """Load run data directly from ORM."""
    with session_context(engine) as session:
        stmt = (
            select(InferRunORM)
            .options(joinedload(InferRunORM.infer_run_output))
            .where(InferRunORM.run_name == run_name)
        )
        infer_run = session.execute(stmt).scalar_one_or_none()

        if infer_run is None:
            raise ValueError(f"Run not found: {run_name}")

        if infer_run.infer_run_output is None:
            raise ValueError(f"Run has no output: {run_name}")

        output = infer_run.infer_run_output
        sample_fingerprint = output.sample_fingerprint

        if not sample_fingerprint:
            raise ValueError(f"Run has no sample fingerprint: {run_name}")

        stmt = (
            select(LLMJudgementORM)
            .options(joinedload(LLMJudgementORM.llm_score))
            .where(LLMJudgementORM.infer_run_output_id == output.id)
            .order_by(LLMJudgementORM.normalized_dataset_judging_sample_id)
        )
        judgements = session.execute(stmt).scalars().all()

        sample_ids = [str(j.normalized_dataset_judging_sample_id) for j in judgements]
        predictions = []
        for j in judgements:
            if j.llm_score is not None:
                predictions.append(j.llm_score.label.value)
            else:
                predictions.append(None)

        return RunData(
            run_name=run_name,
            sample_fingerprint=sample_fingerprint,
            sample_ids=sample_ids,
            predictions=predictions,
        )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_run_group(runs: list[RunData]) -> None:
    """Validate that all runs in a group are comparable."""
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 runs, got {len(runs)}")

    fingerprints = {r.sample_fingerprint for r in runs}
    if len(fingerprints) > 1:
        raise ValueError(
            f"Sample fingerprints don't match across runs:\n"
            + "\n".join(f"  {r.run_name}: {r.sample_fingerprint}" for r in runs)
        )

    reference_ids = runs[0].sample_ids
    for run in runs[1:]:
        if run.sample_ids != reference_ids:
            raise ValueError(
                f"Sample IDs don't match between {runs[0].run_name} and {run.run_name}"
            )

    lengths = [len(r.predictions) for r in runs]
    if len(set(lengths)) > 1:
        raise ValueError(f"Prediction counts don't match: {lengths}")


# ============================================================================
# ANALYSIS
# ============================================================================


@dataclass
class PairwiseKappa:
    """A single pairwise kappa value."""

    run_a: str
    run_b: str
    kappa: float


@dataclass
class SelfAgreementResult:
    """Result of self-agreement analysis for a model."""

    model_label: str
    pairwise_kappas: list[PairwiseKappa]
    n_items: int
    sample_fingerprint: str

    @property
    def kappa_values(self) -> list[float]:
        return [pk.kappa for pk in self.pairwise_kappas]

    @property
    def mean_kappa(self) -> float:
        return np.mean(self.kappa_values)

    @property
    def min_kappa(self) -> float:
        return min(self.kappa_values)

    @property
    def max_kappa(self) -> float:
        return max(self.kappa_values)

    @property
    def range_kappa(self) -> float:
        return self.max_kappa - self.min_kappa


def compute_pairwise_kappa(
    predictions_a: list[int | None], predictions_b: list[int | None]
) -> float:
    """Compute Cohen's kappa between two sets of predictions."""
    # Filter out samples where either prediction is None
    valid_pairs = [
        (a, b) for a, b in zip(predictions_a, predictions_b) if a is not None and b is not None
    ]
    if len(valid_pairs) < 10:
        raise ValueError(f"Too few valid pairs: {len(valid_pairs)}")

    y1, y2 = zip(*valid_pairs)
    return cohen_kappa_score(y1, y2, weights="linear")


def analyze_run_groups(
    run_groups: list[ModelRunGroup], engine
) -> list[SelfAgreementResult]:
    """Analyze pairwise self-agreement for each model group."""
    results = []

    for group in run_groups:
        typer.echo(f"\nAnalyzing {group.model_label}...")

        # Load all runs
        runs: list[RunData] = []
        for run_name in group.run_names:
            try:
                run_data = load_run_data(run_name, engine)
                runs.append(run_data)
                typer.echo(f"  Loaded {run_name}: {len(run_data.predictions)} items")
            except Exception as e:
                typer.echo(f"  Warning: Failed to load {run_name}: {e}", err=True)

        if len(runs) < 2:
            typer.echo(f"  Skipping: need at least 2 runs, got {len(runs)}", err=True)
            continue

        # Validate
        try:
            validate_run_group(runs)
            typer.echo(
                f"  Validation passed: fingerprint={runs[0].sample_fingerprint[:16]}..."
            )
        except ValueError as e:
            typer.echo(f"  Validation failed: {e}", err=True)
            continue

        # Compute pairwise kappas
        pairwise_kappas = []
        for i, j in combinations(range(len(runs)), 2):
            run_a, run_b = runs[i], runs[j]
            kappa = compute_pairwise_kappa(run_a.predictions, run_b.predictions)
            pk = PairwiseKappa(
                run_a=f"Run{i+1}",
                run_b=f"Run{j+1}",
                kappa=kappa,
            )
            pairwise_kappas.append(pk)
            typer.echo(f"  κ({pk.run_a}↔{pk.run_b}) = {kappa:.3f}")

        result = SelfAgreementResult(
            model_label=group.model_label,
            pairwise_kappas=pairwise_kappas,
            n_items=len(runs[0].predictions),
            sample_fingerprint=runs[0].sample_fingerprint,
        )
        results.append(result)
        typer.echo(
            f"  Mean κ = {result.mean_kappa:.3f}, "
            f"Range = {result.min_kappa:.3f} - {result.max_kappa:.3f}"
        )

    return results


# ============================================================================
# PLOTTING
# ============================================================================


def plot_dumbbell_chart(results: list[SelfAgreementResult], output_path: Path):
    """Create dumbbell chart showing range of pairwise kappas per model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(results))
    labels = [r.model_label for r in results]

    for i, result in enumerate(results):
        kappas = result.kappa_values
        min_k, max_k = min(kappas), max(kappas)

        # Draw connecting line (the "dumbbell bar")
        ax.plot(
            [min_k, max_k],
            [i, i],
            color=GRAY_SCALE["medium"],
            linewidth=3,
            solid_capstyle="round",
            zorder=1,
        )

        # Draw all kappa points
        for j, kappa in enumerate(kappas):
            color = [BHT_COLORS["blue"], BHT_COLORS["yellow"], BHT_COLORS["turquoise"]][j % 3]
            ax.scatter(
                kappa,
                i,
                s=120,
                color=color,
                edgecolor="black",
                linewidth=1,
                zorder=2,
            )

        # Add range annotation on the right
        range_val = max_k - min_k
        ax.annotate(
            f"Δ={range_val:.3f}",
            xy=(max_k + 0.02, i),
            va="center",
            fontsize=9,
            color=BHT_COLORS["red"] if range_val > 0.05 else GRAY_SCALE["dark"],
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's κ (pairwise)", fontsize=12, fontweight="bold")
    ax.set_title(PLOT_TITLE + " - Dumbbell Chart", fontsize=14, fontweight="bold")

    # Set x limits with some padding
    all_kappas = [k for r in results for k in r.kappa_values]
    x_min = min(all_kappas) - 0.1
    x_max = max(all_kappas) + 0.15
    ax.set_xlim(x_min, x_max)

    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BHT_COLORS["blue"], edgecolor="black", label="Run1↔Run2"),
        Patch(facecolor=BHT_COLORS["yellow"], edgecolor="black", label="Run1↔Run3"),
        Patch(facecolor=BHT_COLORS["turquoise"], edgecolor="black", label="Run2↔Run3"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nDumbbell chart saved to: {output_path}")
    plt.close()

    copy_figure_to_overleaf(output_path)


def plot_dot_chart_with_jitter(results: list[SelfAgreementResult], output_path: Path):
    """Create dot plot with jitter showing all pairwise kappas."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(len(results))
    labels = [r.model_label for r in results]

    # Jitter settings
    jitter_width = 0.15

    for i, result in enumerate(results):
        kappas = result.kappa_values
        n_points = len(kappas)

        # Create jittered x positions
        if n_points == 1:
            x_jitter = [i]
        else:
            x_jitter = np.linspace(i - jitter_width, i + jitter_width, n_points)

        # Plot points with different colors
        colors = [BHT_COLORS["blue"], BHT_COLORS["yellow"], BHT_COLORS["turquoise"]]
        for j, (x, kappa) in enumerate(zip(x_jitter, kappas)):
            ax.scatter(
                x,
                kappa,
                s=150,
                color=colors[j % len(colors)],
                edgecolor="black",
                linewidth=1.5,
                zorder=2,
            )

        # Add mean line
        mean_k = result.mean_kappa
        ax.plot(
            [i - 0.25, i + 0.25],
            [mean_k, mean_k],
            color=BHT_COLORS["red"],
            linewidth=2,
            linestyle="--",
            zorder=1,
        )

        # Add mean value annotation
        ax.annotate(
            f"μ={mean_k:.3f}",
            xy=(i, mean_k - 0.03),
            ha="center",
            va="top",
            fontsize=9,
            color=BHT_COLORS["red"],
            fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cohen's κ (pairwise)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(PLOT_TITLE + " - Dot Plot", fontsize=14, fontweight="bold")

    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=BHT_COLORS["blue"], edgecolor="black", label="Run1↔Run2"),
        Patch(facecolor=BHT_COLORS["yellow"], edgecolor="black", label="Run1↔Run3"),
        Patch(facecolor=BHT_COLORS["turquoise"], edgecolor="black", label="Run2↔Run3"),
        Line2D([0], [0], color=BHT_COLORS["red"], linestyle="--", linewidth=2, label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nDot plot saved to: {output_path}")
    plt.close()

    copy_figure_to_overleaf(output_path)


def print_summary(results: list[SelfAgreementResult]):
    """Print summary statistics."""
    typer.echo("\n" + "=" * 60)
    typer.echo("SUMMARY: Pairwise Cohen's κ")
    typer.echo("=" * 60)

    for r in results:
        kappas_str = ", ".join([f"{k:.3f}" for k in r.kappa_values])
        typer.echo(
            f"{r.model_label:15s}: κ=[{kappas_str}], "
            f"mean={r.mean_kappa:.3f}, range={r.range_kappa:.3f}"
        )

    typer.echo("-" * 60)
    all_means = [r.mean_kappa for r in results]
    all_ranges = [r.range_kappa for r in results]
    typer.echo(f"Average mean κ:  {np.mean(all_means):.3f}")
    typer.echo(f"Average range:   {np.mean(all_ranges):.3f}")
    typer.echo(f"Max range:       {np.max(all_ranges):.3f}")


# ============================================================================
# MAIN
# ============================================================================

app = typer.Typer()


@app.command()
def main():
    """Compute and plot pairwise Cohen's kappa for self-agreement."""
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dumbbell_path = figures_dir / OUTPUT_DUMBBELL
    dotplot_path = figures_dir / OUTPUT_DOTPLOT

    typer.echo(f"Analyzing {len(RUN_GROUPS)} models for pairwise self-agreement...")

    engine = get_engine()
    results = analyze_run_groups(RUN_GROUPS, engine)

    if not results:
        typer.echo("Error: No results collected", err=True)
        raise typer.Exit(1)

    # Summary
    print_summary(results)

    # Plots
    plot_dumbbell_chart(results, dumbbell_path)
    plot_dot_chart_with_jitter(results, dotplot_path)


if __name__ == "__main__":
    app()
