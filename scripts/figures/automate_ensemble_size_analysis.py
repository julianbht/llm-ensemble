#!/usr/bin/env python3
"""Automate ensemble size vs Cohen's kappa analysis.

This script:
1. Progressively aggregates ensemble runs (1, 1+2, 1+2+3, etc.)
2. Evaluates each aggregated result
3. Outputs a summary of created runs for plotting

The results can then be plotted using scripts/figures/plot_ensemble_size_vs_kappa.py.

Usage:
    python scripts/automate_ensemble_size_analysis.py
"""

import subprocess
import sys
from pathlib import Path
from typing import List

import typer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ordered list of ensemble inference runs to progressively combine
ENSEMBLE_RUNS = [
    "ensemble-4-meta-llama-3.2-3b-instruct-start",
    "ensemble-2-ministral-3b-2515",
    "ensemble-1-google-gemma-3-4b-it",
    "ensemble-3-phi-4-multimodal-instruct",
    "ensemble-5-ui-tars-1.5-7b-start",
    "ensemble-6-cohere-command-r7b-12-2024-start",
    "ensemble-7-qwen3-8b-start",
]

# Aggregation strategy to use
AGGREGATION_STRATEGY = "majority_vote_average"

# I/O configuration names
AGGREGATE_IO_CFG = "db_to_db"
EVALUATE_IO_CFG = "db_aggregate_to_json"

# Run type (official vs test)
RUN_TYPE = "official"

# ============================================================================


app = typer.Typer()


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status.

    Args:
        cmd: Command to run as list of strings
        description: Human-readable description for logging

    Returns:
        True if command succeeded, False otherwise
    """
    typer.echo(f"\n{'='*70}")
    typer.echo(f"Running: {description}")
    typer.echo(f"Command: {' '.join(cmd)}")
    typer.echo(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)

    if result.returncode != 0:
        typer.echo(
            f"\nERROR: {description} failed with code {result.returncode}", err=True
        )
        return False

    typer.echo(f"\nSUCCESS: {description} completed")
    return True


def run_aggregate(
    ensemble_size: int,
    input_runs: List[str],
    python_path: Path,
    aggregate_cli_path: Path,
) -> str:
    """Run aggregate CLI for given ensemble size.

    Args:
        ensemble_size: Size of ensemble (for naming)
        input_runs: List of inference run names to aggregate
        python_path: Path to Python interpreter
        aggregate_cli_path: Path to aggregate CLI

    Returns:
        Run name for the aggregated result
    """
    run_name = f"{ensemble_size}-ensemble-size-analysis"

    # Build command
    cmd = [
        str(python_path),
        str(aggregate_cli_path),
        "--aggregation-strategy",
        AGGREGATION_STRATEGY,
        "--io-cfg",
        AGGREGATE_IO_CFG,
    ]

    # Add input runs
    for run in input_runs:
        cmd.extend(["-i", run])

    # Add run metadata
    cmd.extend(
        [
            "--run-name",
            run_name,
            "--notes",
            f"Ensemble size {ensemble_size} analysis: aggregating {len(input_runs)} models with {AGGREGATION_STRATEGY}",
        ]
    )

    if RUN_TYPE == "official":
        cmd.append("--official")

    # Run command
    success = run_command(
        cmd, f"Aggregate: ensemble size {ensemble_size} ({len(input_runs)} models)"
    )

    if not success:
        raise typer.Exit(1)

    return run_name


def run_evaluate(
    aggregate_run_name: str,
    python_path: Path,
    evaluate_cli_path: Path,
):
    """Run evaluate CLI for aggregated result.

    Args:
        aggregate_run_name: Name of the aggregate run to evaluate
        python_path: Path to Python interpreter
        evaluate_cli_path: Path to evaluate CLI
    """
    # Build command
    cmd = [
        str(python_path),
        str(evaluate_cli_path),
        "--io-cfg",
        EVALUATE_IO_CFG,
        "--input",
        aggregate_run_name,
        "--run-name",
        aggregate_run_name,
        "--notes",
        f"Evaluation of {aggregate_run_name}",
    ]

    if RUN_TYPE == "official":
        cmd.append("--official")

    # Run command
    success = run_command(cmd, f"Evaluate: {aggregate_run_name}")

    if not success:
        raise typer.Exit(1)


@app.command()
def main(
    max_ensemble_size: int = typer.Option(
        None, help="Maximum ensemble size to analyze (default: all available runs)"
    ),
    skip_aggregate: bool = typer.Option(
        False, help="Skip aggregate step (useful if already aggregated)"
    ),
    skip_evaluate: bool = typer.Option(
        False, help="Skip evaluate step (useful for testing)"
    ),
):
    """Automate ensemble size analysis: aggregate + evaluate progressively.

    This will create runs for ensemble sizes 1 through N, where N is either
    max_ensemble_size or the number of available runs.
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    python_path = project_root / ".venv" / "bin" / "python3"
    aggregate_cli_path = (
        project_root
        / "src"
        / "llm_ensemble"
        / "aggregate"
        / "adapters"
        / "driving"
        / "aggregate_cli.py"
    )
    evaluate_cli_path = (
        project_root
        / "src"
        / "llm_ensemble"
        / "evaluate"
        / "adapters"
        / "driving"
        / "evaluate_cli.py"
    )

    # Validate paths
    if not python_path.exists():
        typer.echo(f"ERROR: Python interpreter not found: {python_path}", err=True)
        typer.echo("Please ensure virtual environment is created at .venv/", err=True)
        raise typer.Exit(1)

    # Determine max ensemble size
    num_runs = len(ENSEMBLE_RUNS)
    if max_ensemble_size is None:
        max_ensemble_size = num_runs
    else:
        max_ensemble_size = min(max_ensemble_size, num_runs)

    typer.echo(f"\n{'#'*70}")
    typer.echo("# ENSEMBLE SIZE ANALYSIS AUTOMATION")
    typer.echo(f"{'#'*70}")
    typer.echo(f"\nConfiguration:")
    typer.echo(f"  Available runs: {num_runs}")
    typer.echo(f"  Max ensemble size: {max_ensemble_size}")
    typer.echo(f"  Aggregation strategy: {AGGREGATION_STRATEGY}")
    typer.echo(f"  Run type: {RUN_TYPE}")
    typer.echo(f"  Skip aggregate: {skip_aggregate}")
    typer.echo(f"  Skip evaluate: {skip_evaluate}")
    typer.echo(f"\nEnsemble runs to combine:")
    for i, run in enumerate(ENSEMBLE_RUNS[:max_ensemble_size], 1):
        typer.echo(f"  {i}. {run}")

    # Confirm before proceeding
    if not typer.confirm("\nProceed with automation?"):
        typer.echo("Aborted.")
        raise typer.Exit(0)

    # Track created runs for summary
    created_runs = []

    # Process each ensemble size progressively
    for ensemble_size in range(1, max_ensemble_size + 1):
        input_runs = ENSEMBLE_RUNS[:ensemble_size]

        typer.echo(f"\n\n{'#'*70}")
        typer.echo(f"# PROCESSING ENSEMBLE SIZE {ensemble_size}")
        typer.echo(f"{'#'*70}")

        # Aggregate
        if not skip_aggregate:
            aggregate_run_name = run_aggregate(
                ensemble_size,
                input_runs,
                python_path,
                aggregate_cli_path,
            )
        else:
            aggregate_run_name = f"{ensemble_size}-ensemble-size-analysis"
            typer.echo(
                f"\nSkipping aggregate (using existing run: {aggregate_run_name})"
            )

        # Evaluate
        if not skip_evaluate:
            run_evaluate(
                aggregate_run_name,
                python_path,
                evaluate_cli_path,
            )
        else:
            typer.echo(f"\nSkipping evaluate for: {aggregate_run_name}")

        created_runs.append(aggregate_run_name)

    # Print summary
    typer.echo(f"\n\n{'#'*70}")
    typer.echo("# AUTOMATION COMPLETE")
    typer.echo(f"{'#'*70}")
    typer.echo(f"\nCreated {len(created_runs)} runs:")
    for i, run_name in enumerate(created_runs, 1):
        typer.echo(f"  {i}. {run_name}")

    typer.echo(f"\nNext steps:")
    typer.echo(
        f"  1. Verify evaluate runs exist in: artifacts/runs/evaluate/{RUN_TYPE}/"
    )
    typer.echo(
        f"  2. Update RUNS constant in scripts/figures/plot_ensemble_size_vs_kappa.py"
    )
    typer.echo(f"  3. Run: python scripts/figures/plot_ensemble_size_vs_kappa.py")
    typer.echo(f"\nExample RUNS constant:")
    typer.echo("  RUNS = [")
    for run_name in created_runs:
        typer.echo(f'      "{run_name}",')
    typer.echo("  ]")


if __name__ == "__main__":
    app()
