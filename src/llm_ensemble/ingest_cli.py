from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

from llm_ensemble.ingest.adapters import load_adapter
from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.libs.config import load_dataset_config
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.logging.logger import get_logger

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


def _json_dumps(obj: JudgingExample) -> str:
    return obj.model_dump_json()


@app.command("ingest")
def ingest(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset ID to ingest (e.g., 'llm-judge-2024')"
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

    # Load dataset config
    try:
        config = load_dataset_config(dataset)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e))

    # Use data_dir override if provided, otherwise use config default
    actual_data_dir = data_dir if data_dir is not None else config.data_dir

    # Verify data directory exists
    if not actual_data_dir.exists():
        raise typer.BadParameter(f"Data directory does not exist: {actual_data_dir}")

    # Load the adapter dynamically
    try:
        iter_examples = load_adapter(config.adapter)
    except (ImportError, AttributeError) as e:
        raise typer.BadParameter(f"Failed to load adapter '{config.adapter}': {e}")

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(config.dataset_id)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="ingest", official=official)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "samples.ndjson"

    # Set up log file if requested
    log_file_handle = None
    if save_logs:
        log_file_path = run_dir / "run.log"
        log_file_handle = open(log_file_path, "w", encoding="utf-8")

    # Initialize logger
    logger = get_logger("ingest", run_id=run_id, log_file=log_file_handle)

    logger.info(
        "Starting ingest",
        dataset_id=config.dataset_id,
        adapter=config.adapter,
        data_dir=str(actual_data_dir),
        limit=limit,
    )
    logger.info("Run directory", path=str(run_dir))
    logger.info("Output file", path=str(output_file))

    # Process examples
    count = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as sink:
        for ex in iter_examples(actual_data_dir):
            sink.write(_json_dumps(ex) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    logger.info("Ingest complete", total_examples=count)

    # Write manifest
    write_manifest(
        run_dir=run_dir,
        cli_name="ingest",
        cli_args={
            "dataset_id": config.dataset_id,
            "adapter": config.adapter,
            "data_dir": str(actual_data_dir),
            "limit": limit,
        },
        metadata={
            "sample_count": count,
            "output_file": str(output_file),
            "dataset_version": config.version,
        },
        official=official,
        notes=notes,
    )

    logger.info("Manifest written", path=str(run_dir / "manifest.json"))

    # Close log file if opened
    if log_file_handle is not None:
        logger.info("Logs saved", path=str(run_dir / "run.log"))
        log_file_handle.close()


if __name__ == "__main__":
    app()
