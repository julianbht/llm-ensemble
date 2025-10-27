from __future__ import annotations
from typing import Optional

import typer

from llm_ensemble.ingest.orchestrator import run_ingest
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.utils.config_overrides import parse_overrides

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


@app.command("ingest")
def ingest(
    io_format: str = typer.Option(
        ..., "--io", help="I/O format config name (e.g., 'llm_judge_ingest' for configs/io/llm_judge_ingest.yaml)"
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
    override: list[str] = typer.Option(
        [],
        "--override",
        "-O",
        help="Override config values (format: key=value, e.g., 'data_dir=/custom/path'). Can be specified multiple times."
    ),
):
    """Normalize a raw IR dataset into JudgingExample records.

    Writes output to artifacts/runs/<run_id>/samples.<format> with manifest.

    All behavior is explicitly configured via I/O config files - no implicit defaults.

    Examples:
        # Basic usage
        ingest --io llm_judge_ingest --limit 100

        # Override data directory
        ingest --io llm_judge_ingest --override data_dir=/custom/path

        # Multiple overrides
        ingest --io llm_judge_ingest --override data_dir=/data --override dataset_id=custom-v2

        See config files in configs/io/ for available fields.
        Overrides are tracked in manifest for reproducibility.
    """
    try:
        # Parse overrides
        config_overrides = parse_overrides(override) if override else {}

        run_ingest(
            io_format=io_format,
            run_id=run_id,
            limit=limit,
            save_logs=save_logs,
            official=official,
            notes=notes,
            config_overrides=config_overrides,
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
