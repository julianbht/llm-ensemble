from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

from llm_ensemble.ingest.adapters.llm_judge import iter_examples
from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.libs.logging import configure_logging
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.runtime.git_utils import get_git_info

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


def _json_dumps(obj: JudgingExample) -> str:
    return obj.model_dump_json()


@app.command("ingest")
def ingest(
    dataset: str = typer.Option(
        ..., "--adapter", "-a", help="Dataset adapter that reads in raw data and produces JudgingExample records"
    ),
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-i", exists=True, file_okay=False, readable=True,
        help="Path to directory containing raw dataset files",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Custom run ID (auto-generates if not provided)"
    ),
    limit: Optional[int] = typer.Option(None, help="Process at most N examples"),
):
    """Normalize a raw IR dataset into JudgingExample NDJSON records.

    Writes output to artifacts/runs/<run_id>/samples.ndjson with manifest.

    Example:
        ingest --adapter llm-judge --data-dir ./data --limit 100
    """

    dataset = dataset.lower().strip()
    if dataset not in {"llm-judge", "llm-judge-2024"}:
        raise typer.BadParameter("Unsupported dataset. Try: llm-judge")

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(dataset)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="ingest")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "samples.ndjson"
    log_file = run_dir / "logs.jsonl"

    # Configure structured logging
    git_info = get_git_info()
    logger = configure_logging(
        cli_name="ingest",
        run_id=run_id,
        log_file=log_file,
        git_sha=git_info.get("git_sha"),
    )

    typer.echo(f"Run ID: {run_id}", err=True)
    typer.echo(f"Output: {output_file}", err=True)

    logger.info(
        "ingest_started",
        dataset=dataset,
        data_dir=str(data_dir),
        limit=limit,
        output_file=str(output_file),
    )

    # Process examples
    count = 0
    try:
        with output_file.open("w", encoding="utf-8", newline="\n") as sink:
            for ex in iter_examples(data_dir):
                sink.write(_json_dumps(ex) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break

        typer.echo(f"Wrote {count} examples", err=True)
        logger.info("ingest_completed", sample_count=count)

    except Exception as e:
        logger.error("ingest_failed", error=str(e), error_type=type(e).__name__)
        typer.echo(f"Error during ingest: {e}", err=True)
        raise

    # Write manifest
    write_manifest(
        run_dir=run_dir,
        cli_name="ingest",
        cli_args={
            "dataset": dataset,
            "data_dir": str(data_dir),
            "limit": limit,
        },
        metadata={
            "sample_count": count,
            "output_file": str(output_file),
            "log_file": str(log_file),
        },
    )

    typer.echo(f"Manifest: {run_dir / 'manifest.json'}", err=True)


if __name__ == "__main__":
    app()
