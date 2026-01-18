#!/usr/bin/env python3
"""Plot Krippendorff's alpha for self-agreement across repeated runs.

Measures how consistently models judge the same items across multiple identical
runs. Each run is treated as a separate "rater" and Krippendorff's alpha
measures their agreement.

High alpha = model is consistent/deterministic
Low alpha = model has high variance in judgements

Usage:
    python scripts/figures/plot_self_agreement_krippendorff.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

import numpy as np
import matplotlib.pyplot as plt
import krippendorff
import typer
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from thesis_colors import BHT_COLORS
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

OUTPUT_FILENAME = "self_agreement_krippendorff.svg"
PLOT_TITLE = "Model Self-Agreement (Krippendorff's α)"


# ============================================================================
# DATA LOADING
# ============================================================================


@dataclass
class RunData:
    """Data loaded from a single infer run."""

    run_name: str
    sample_fingerprint: str
    sample_ids: list[str]  # Ordered list of sample UUIDs
    predictions: list[int | None]  # Ordered predictions matching sample_ids


def load_run_data(run_name: str, engine) -> RunData:
    """Load run data directly from ORM."""
    with session_context(engine) as session:
        # Get infer run with output
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

        # Load judgements ordered by sample ID for deterministic ordering
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

    # Check fingerprints match
    fingerprints = {r.sample_fingerprint for r in runs}
    if len(fingerprints) > 1:
        raise ValueError(
            f"Sample fingerprints don't match across runs:\n"
            + "\n".join(f"  {r.run_name}: {r.sample_fingerprint}" for r in runs)
        )

    # Check sample IDs match in order
    reference_ids = runs[0].sample_ids
    for run in runs[1:]:
        if run.sample_ids != reference_ids:
            raise ValueError(
                f"Sample IDs don't match between {runs[0].run_name} and {run.run_name}"
            )

    # Check prediction counts match
    lengths = [len(r.predictions) for r in runs]
    if len(set(lengths)) > 1:
        raise ValueError(f"Prediction counts don't match: {lengths}")


# ============================================================================
# ANALYSIS
# ============================================================================


@dataclass
class SelfAgreementResult:
    """Result of self-agreement analysis for a model."""

    model_label: str
    alpha: float
    n_runs: int
    n_items: int
    sample_fingerprint: str


def compute_krippendorff_alpha(run_predictions: list[list[int | None]]) -> float:
    """Compute Krippendorff's alpha across multiple runs."""
    n_runs = len(run_predictions)
    n_items = len(run_predictions[0])

    reliability_data = np.zeros((n_runs, n_items), dtype=np.float64)
    for run_idx, predictions in enumerate(run_predictions):
        for item_idx, pred in enumerate(predictions):
            reliability_data[run_idx, item_idx] = pred if pred is not None else np.nan

    return krippendorff.alpha(
        reliability_data=reliability_data,
        level_of_measurement="ordinal",
        value_domain=[0, 1, 2, 3],
    )


def analyze_run_groups(
    run_groups: list[ModelRunGroup], engine
) -> list[SelfAgreementResult]:
    """Analyze self-agreement for each model group."""
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

        # Compute alpha
        predictions = [r.predictions for r in runs]
        alpha = compute_krippendorff_alpha(predictions)

        result = SelfAgreementResult(
            model_label=group.model_label,
            alpha=alpha,
            n_runs=len(runs),
            n_items=len(runs[0].predictions),
            sample_fingerprint=runs[0].sample_fingerprint,
        )
        results.append(result)
        typer.echo(
            f"  Krippendorff's α = {alpha:.3f} ({result.n_runs} runs, {result.n_items} items)"
        )

    return results


# ============================================================================
# PLOTTING
# ============================================================================


def print_latex_table(results: list[SelfAgreementResult], output_path: Path):
    """Generate LaTeX table of self-agreement scores."""
    typer.echo("\n" + "=" * 60)
    typer.echo("LATEX TABLE")
    typer.echo("=" * 60)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Model Self-Agreement Across Repeated Runs (Krippendorff's $\alpha$)}",
        r"  \label{tab:self-agreement-krippendorff}",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Model & Krippendorff's $\alpha$ & Runs \\",
        r"    \midrule",
    ]

    for r in results:
        latex_lines.append(f"    {r.model_label} & {r.alpha:.3f} & {r.n_runs} \\\\")

    latex_lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )

    latex_content = "\n".join(latex_lines)
    typer.echo(latex_content)

    # Save to file
    table_path = output_path.with_suffix(".tex")
    table_path.write_text(latex_content)
    typer.echo(f"\nLaTeX table saved to: {table_path}")


def plot_bar_chart(results: list[SelfAgreementResult], output_path: Path):
    """Create bar chart of self-agreement scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [r.model_label for r in results]
    alphas = [r.alpha for r in results]

    bars = ax.bar(
        labels, alphas, color=BHT_COLORS["blue"], edgecolor="black", linewidth=0.5
    )

    for bar, alpha in zip(bars, alphas):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            alpha + 0.01,
            f"{alpha:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Krippendorff's α", fontsize=12, fontweight="bold")
    ax.set_title(PLOT_TITLE, fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()

    copy_figure_to_overleaf(output_path)


# ============================================================================
# MAIN
# ============================================================================

app = typer.Typer()


@app.command()
def main():
    """Compute and plot self-agreement (Krippendorff's alpha) across repeated runs."""
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / OUTPUT_FILENAME

    typer.echo(f"Analyzing {len(RUN_GROUPS)} models for self-agreement...")

    engine = get_engine()
    results = analyze_run_groups(RUN_GROUPS, engine)

    if not results:
        typer.echo("Error: No results collected", err=True)
        raise typer.Exit(1)

    # Summary
    typer.echo("\n" + "=" * 60)
    typer.echo("SUMMARY")
    typer.echo("=" * 60)
    avg_alpha = np.mean([r.alpha for r in results])
    typer.echo(f"Average Krippendorff's α: {avg_alpha:.3f}")
    typer.echo(
        f"Range: {min(r.alpha for r in results):.3f} - {max(r.alpha for r in results):.3f}"
    )

    # Generate LaTeX table
    print_latex_table(results, output_path)

    # Plot
    plot_bar_chart(results, output_path)


if __name__ == "__main__":
    app()
