from __future__ import annotations
from typing import Optional

import typer

from llm_ensemble.ingest.orchestrator import run_ingest
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.cli.common_params import InputPath, IoFormat, RunId, SaveLogs, Official, Notes

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


@app.command("ingest")
def ingest(
    input_path: InputPath,
    io_format: IoFormat,
    limit: Optional[int] = typer.Option(None, help="Process at most N examples"),
    run_id: RunId = None,
    save_logs: SaveLogs = False,
    official: Official = False,
    notes: Notes = None,
):
    """Normalize a raw IR dataset into JudgingExample records.

    Writes output to artifacts/runs/<run_id>/normalized_dataset.ndjson with manifest.

    All behavior is explicitly configured via I/O config files - no implicit defaults.

    Examples:
        # Basic usage
        ingest --input data/llm-judge-2024 --io llm_judge_challenge --limit 100

        # Official run with notes
        ingest -i data/llm-judge-2024 --io llm_judge_challenge --official --notes "Baseline dataset"
    """
    run_ingest(
        input_path=input_path,
        io_format=io_format,
        run_id=run_id,
        limit=limit,
        save_logs=save_logs,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()
