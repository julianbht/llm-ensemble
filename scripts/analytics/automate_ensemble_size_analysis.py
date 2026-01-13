#!/usr/bin/env python3
"""Automate ensemble size vs Cohen's kappa analysis with multiple strategies.

This script:
1. Progressively aggregates ensemble runs (1, 1+2, 1+2+3, etc.)
2. Uses multiple aggregation strategies for each ensemble size
3. Evaluates each aggregated result
4. Outputs a summary of created runs for plotting

The results can then be plotted using scripts/figures/plot_ensemble_size_vs_kappa.py.

Usage:
    python scripts/figures/automate_ensemble_size_analysis.py
"""

import subprocess
from pathlib import Path
from typing import List, Dict

import typer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ordered list of ensemble inference runs to progressively combine
ENSEMBLE_RUNS = [
    "ensemble-7-qwen3-8b-start",
    "ensemble-6-cohere-command-r7b-12-2024-start",
    "ensemble-5-ui-tars-1.5-7b-start",
    "ensemble-3-phi-4-multimodal-instruct",
    "ensemble-1-google-gemma-3-4b-it",
    "ensemble-2-ministral-3b-2515",
    "ensemble-4-meta-llama-3.2-3b-instruct-start",
]

# Aggregation strategies to compare (all will be plotted together)
AGGREGATION_STRATEGIES = [
    "average_vote",
    "majority_vote_average",
    "majority_vote_random",
]

# I/O configuration names
AGGREGATE_IO_CFG = "db_to_db"
EVALUATE_IO_CFG = "db_aggregate_to_json"

# Run type (official vs test)
RUN_TYPE = "official"

# Run name prefix/suffix (optional, for distinguishing different experiment runs)
# Examples: "order1", "random", "best-first", etc.
RUN_PREFIX = ""
RUN_SUFFIX = "params_big_to_small_v4"

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


def check_directory_exists(
    run_name: str, run_type: str, project_root: Path, check_aggregate: bool = True
) -> None:
    """Check if run directory already exists and raise error if it does.

    Args:
        run_name: Name of the run
        run_type: Type of run (official, test)
        project_root: Project root path
        check_aggregate: If True, check aggregate dir, else check evaluate dir

    Raises:
        typer.Exit: If directory already exists
    """
    if check_aggregate:
        dir_path = (
            project_root / "artifacts" / "runs" / "aggregate" / run_type / run_name
        )
    else:
        dir_path = (
            project_root / "artifacts" / "runs" / "evaluate" / run_type / run_name
        )

    if dir_path.exists():
        typer.echo(
            f"\nERROR: Run directory already exists: {dir_path}",
            err=True,
        )
        typer.echo(
            "Please use a different RUN_PREFIX or RUN_SUFFIX to avoid overwriting existing runs.",
            err=True,
        )
        raise typer.Exit(1)


def build_run_name(ensemble_size: int, strategy: str) -> str:
    """Build run name from components.

    Args:
        ensemble_size: Size of ensemble
        strategy: Aggregation strategy name

    Returns:
        Formatted run name
    """
    parts = []
    if RUN_PREFIX:
        parts.append(RUN_PREFIX)
    parts.append(f"{ensemble_size}-ensemble-size-analysis")
    parts.append(strategy)
    if RUN_SUFFIX:
        parts.append(RUN_SUFFIX)
    return "-".join(parts)


def run_aggregate(
    ensemble_size: int,
    strategy: str,
    input_runs: List[str],
    python_path: Path,
    aggregate_cli_path: Path,
    project_root: Path,
) -> str:
    """Run aggregate CLI for given ensemble size and strategy.

    Args:
        ensemble_size: Size of ensemble (for naming)
        strategy: Aggregation strategy name
        input_runs: List of inference run names to aggregate
        python_path: Path to Python interpreter
        aggregate_cli_path: Path to aggregate CLI
        project_root: Project root path

    Returns:
        Run name for the aggregated result
    """
    run_name = build_run_name(ensemble_size, strategy)

    # Check if directory already exists
    check_directory_exists(run_name, RUN_TYPE, project_root, check_aggregate=True)

    # Build command
    cmd = [
        str(python_path),
        str(aggregate_cli_path),
        "--aggregation-strategy",
        strategy,
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
            f"Ensemble size {ensemble_size} analysis: aggregating {len(input_runs)} models with {strategy}",
        ]
    )

    if RUN_TYPE == "official":
        cmd.append("--official")

    # Run command
    success = run_command(
        cmd,
        f"Aggregate: ensemble_size={ensemble_size}, strategy={strategy}",
    )

    if not success:
        raise typer.Exit(1)

    return run_name


def run_evaluate(
    aggregate_run_name: str,
    python_path: Path,
    evaluate_cli_path: Path,
    project_root: Path,
):
    """Run evaluate CLI for aggregated result.

    Args:
        aggregate_run_name: Name of the aggregate run to evaluate
        python_path: Path to Python interpreter
        evaluate_cli_path: Path to evaluate CLI
        project_root: Project root path
    """
    # Check if directory already exists
    check_directory_exists(
        aggregate_run_name, RUN_TYPE, project_root, check_aggregate=False
    )

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
    """Automate ensemble size analysis with multiple aggregation strategies.

    This will create runs for:
    - Ensemble sizes 1 through N (where N is max_ensemble_size or all runs)
    - All configured aggregation strategies

    Total runs created: N × len(AGGREGATION_STRATEGIES)
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

    total_runs = max_ensemble_size * len(AGGREGATION_STRATEGIES)

    typer.echo(f"\n{'#'*70}")
    typer.echo("# ENSEMBLE SIZE ANALYSIS AUTOMATION (MULTI-STRATEGY)")
    typer.echo(f"{'#'*70}")
    typer.echo(f"\nConfiguration:")
    typer.echo(f"  Available inference runs: {num_runs}")
    typer.echo(f"  Max ensemble size: {max_ensemble_size}")
    typer.echo(f"  Aggregation strategies: {len(AGGREGATION_STRATEGIES)}")
    for strategy in AGGREGATION_STRATEGIES:
        typer.echo(f"    - {strategy}")
    typer.echo(f"  Run type: {RUN_TYPE}")
    typer.echo(f"  Run prefix: '{RUN_PREFIX}' (empty = none)")
    typer.echo(f"  Run suffix: '{RUN_SUFFIX}' (empty = none)")
    typer.echo(f"  Skip aggregate: {skip_aggregate}")
    typer.echo(f"  Skip evaluate: {skip_evaluate}")
    typer.echo(f"\n  TOTAL RUNS TO CREATE: {total_runs}")
    typer.echo(f"\nEnsemble runs to combine:")
    for i, run in enumerate(ENSEMBLE_RUNS[:max_ensemble_size], 1):
        typer.echo(f"  {i}. {run}")

    # Confirm before proceeding
    if not typer.confirm("\nProceed with automation?"):
        typer.echo("Aborted.")
        raise typer.Exit(0)

    # Track created runs by strategy for summary
    created_runs_by_strategy: Dict[str, List[str]] = {
        strategy: [] for strategy in AGGREGATION_STRATEGIES
    }

    # Process each strategy and ensemble size
    for strategy in AGGREGATION_STRATEGIES:
        typer.echo(f"\n\n{'#'*70}")
        typer.echo(f"# PROCESSING STRATEGY: {strategy}")
        typer.echo(f"{'#'*70}")

        for ensemble_size in range(1, max_ensemble_size + 1):
            input_runs = ENSEMBLE_RUNS[:ensemble_size]

            typer.echo(f"\n{'='*70}")
            typer.echo(f"Strategy: {strategy} | Ensemble size: {ensemble_size}")
            typer.echo(f"{'='*70}")

            # Aggregate
            if not skip_aggregate:
                aggregate_run_name = run_aggregate(
                    ensemble_size,
                    strategy,
                    input_runs,
                    python_path,
                    aggregate_cli_path,
                    project_root,
                )
            else:
                aggregate_run_name = build_run_name(ensemble_size, strategy)
                typer.echo(
                    f"\nSkipping aggregate (using existing run: {aggregate_run_name})"
                )

            # Evaluate
            if not skip_evaluate:
                run_evaluate(
                    aggregate_run_name,
                    python_path,
                    evaluate_cli_path,
                    project_root,
                )
            else:
                typer.echo(f"\nSkipping evaluate for: {aggregate_run_name}")

            created_runs_by_strategy[strategy].append(aggregate_run_name)

    # Print summary
    typer.echo(f"\n\n{'#'*70}")
    typer.echo("# AUTOMATION COMPLETE")
    typer.echo(f"{'#'*70}")
    typer.echo(
        f"\nCreated {total_runs} runs across {len(AGGREGATION_STRATEGIES)} strategies:\n"
    )

    for strategy in AGGREGATION_STRATEGIES:
        runs = created_runs_by_strategy[strategy]
        typer.echo(f"\n{strategy} ({len(runs)} runs):")
        for i, run_name in enumerate(runs, 1):
            typer.echo(f"  {i}. {run_name}")

    typer.echo(f"\n\nNext steps:")
    typer.echo(
        f"  1. Verify evaluate runs exist in: artifacts/runs/evaluate/{RUN_TYPE}/"
    )
    typer.echo(
        f"  2. Update RUNS_BY_STRATEGY in scripts/figures/plot_ensemble_size_vs_kappa.py"
    )
    typer.echo(f"  3. Run: python scripts/figures/plot_ensemble_size_vs_kappa.py")

    typer.echo(f"\n\nExample RUNS_BY_STRATEGY constant:")
    typer.echo("RUNS_BY_STRATEGY = {")
    for strategy in AGGREGATION_STRATEGIES:
        runs = created_runs_by_strategy[strategy]
        typer.echo(f'    "{strategy}": [')
        for run_name in runs:
            typer.echo(f'        (SIZE, "{run_name}"),')
        typer.echo("    ],")
    typer.echo("}")


if __name__ == "__main__":
    app()
