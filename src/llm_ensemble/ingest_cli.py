from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

from llm_ensemble.ingest.orchestrator import run_ingest
from llm_ensemble.libs.runtime.env import load_runtime_config

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


@app.command("ingest")
def ingest(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset ID to ingest (e.g., 'llm-judge-2024') specified in dataset config"
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", "-i", exists=True, file_okay=False, readable=True,
        help="Override data directory from config (defaults to config value)",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Custom run ID (auto-generates if not provided)"
    ),
    limit: Optional[int] = typer.Option(None, help="Process at most N examples"),
    save_logs: bool = typer.Option(
        False, "--save-logs", help="Save logs to run.log file in run directory"
    ),
    official: bool = typer.Option(
        False, "--official", help="Mark as official run (saved to official/ subdirectory for git tracking)"
    ),
    notes: Optional[str] = typer.Option(
        None, "--notes", help="Notes about this run (experiment purpose, hypothesis, etc.)"
    ),
):
    """Normalize a raw IR dataset into JudgingExample NDJSON records.

    Writes output to artifacts/runs/<run_id>/samples.ndjson with manifest.

    Example:
        ingest --dataset llm-judge-2024 --limit 100
        ingest --dataset llm-judge-2024 --data-dir /custom/path
    """
    try:
        run_ingest(
            dataset=dataset,
            data_dir=data_dir,
            run_id=run_id,
            limit=limit,
            save_logs=save_logs,
            official=official,
            notes=notes,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except (ImportError, AttributeError) as e:
        typer.echo(f"Error: Failed to load adapter: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
