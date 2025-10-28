from __future__ import annotations
from typing import Optional

import typer

from llm_ensemble.ingest.orchestrator import run_ingest
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.utils.config_overrides import parse_overrides
from llm_ensemble.libs.cli.common_params import IoFormat, RunId, SaveLogs, Official, Notes, Override

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


@app.command("ingest")
def ingest(
    io_format: IoFormat,
    limit: Optional[int] = typer.Option(None, help="Process at most N examples"),
    run_id: RunId = None,
    save_logs: SaveLogs = False,
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
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


if __name__ == "__main__":
    app()
