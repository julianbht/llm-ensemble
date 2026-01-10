#!/usr/bin/env python3
"""Plot ensemble size vs Cohen's kappa from evaluate runs.

This script reads evaluate_run.json files from multiple runs and creates
a plot showing how Cohen's kappa varies with ensemble size.

Usage:
    python scripts/analyze/plot_ensemble_size_vs_kappa.py

    Or customize the runs to analyze:
    python scripts/analyze/plot_ensemble_size_vs_kappa.py --runs run1,run2,run3
"""

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import typer


app = typer.Typer()


def extract_ensemble_size_from_name(run_name: str) -> int:
    """Extract ensemble size from run name like '2-ensemble-first-figure-check'.

    Assumes the run name starts with the ensemble size as a digit.
    """
    # Try to extract the first digit(s) from the run name
    parts = run_name.split('-')
    if parts and parts[0].isdigit():
        return int(parts[0])
    raise ValueError(f"Cannot extract ensemble size from run name: {run_name}")


def read_evaluate_run(run_path: Path) -> dict:
    """Read and parse evaluate_run.json file."""
    with open(run_path / "evaluate_run.json", 'r') as f:
        return json.load(f)


def extract_cohens_kappa(evaluate_run: dict) -> float:
    """Extract Cohen's kappa value from metric results."""
    for metric in evaluate_run["metric_results"]:
        if metric["name"] == "cohens_kappa":
            return metric["value"]
    raise ValueError("Cohen's kappa not found in metric results")


def collect_data(run_names: List[str], evaluate_runs_base: Path) -> List[Tuple[int, float]]:
    """Collect (ensemble_size, cohens_kappa) pairs from evaluate runs.

    Returns:
        List of (ensemble_size, cohens_kappa) tuples sorted by ensemble size
    """
    data = []

    for run_name in run_names:
        run_path = evaluate_runs_base / run_name
        if not run_path.exists():
            typer.echo(f"Warning: Run path does not exist: {run_path}")
            continue

        evaluate_run = read_evaluate_run(run_path)
        ensemble_size = extract_ensemble_size_from_name(evaluate_run["run_name"])
        cohens_kappa = extract_cohens_kappa(evaluate_run)

        data.append((ensemble_size, cohens_kappa))
        typer.echo(f"Found: ensemble_size={ensemble_size}, cohens_kappa={cohens_kappa:.4f} ({run_name})")

    return sorted(data, key=lambda x: x[0])


def plot_ensemble_size_vs_kappa(data: List[Tuple[int, float]], output_path: Path):
    """Create and save plot of ensemble size vs Cohen's kappa."""
    ensemble_sizes = [d[0] for d in data]
    kappas = [d[1] for d in data]

    plt.figure(figsize=(10, 6))
    plt.plot(ensemble_sizes, kappas, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Ensemble Size', fontsize=12)
    plt.ylabel("Cohen's Kappa", fontsize=12)
    plt.title("Ensemble Size vs Cohen's Kappa", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(ensemble_sizes)  # Show all ensemble sizes on x-axis

    # Add value labels on points
    for size, kappa in data:
        plt.annotate(f'{kappa:.3f}',
                    xy=(size, kappa),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    typer.echo(f"\nPlot saved to: {output_path}")
    plt.close()


@app.command()
def main(
    runs: str = typer.Option(
        "2-ensemble-first-figure-check,3-ensemble-first-figure-check",
        help="Comma-separated list of run names"
    ),
    run_type: str = typer.Option(
        "test",
        help="Run type directory"
    ),
    output: str = typer.Option(
        "ensemble_size_vs_kappa.png",
        help="Output filename"
    ),
):
    """Plot ensemble size vs Cohen's kappa from evaluate runs."""
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    evaluate_runs_base = project_root / "artifacts" / "runs" / "evaluate" / run_type
    figures_dir = project_root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output

    # Parse run names
    run_names = [name.strip() for name in runs.split(',')]

    typer.echo(f"Analyzing {len(run_names)} evaluate runs from: {evaluate_runs_base}\n")

    # Collect data
    data = collect_data(run_names, evaluate_runs_base)

    if not data:
        typer.echo("Error: No data collected. Check that run paths exist and contain valid data.")
        raise typer.Exit(1)

    # Create plot
    plot_ensemble_size_vs_kappa(data, output_path)


if __name__ == "__main__":
    app()
