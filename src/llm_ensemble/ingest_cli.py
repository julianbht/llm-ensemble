from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

from llm_ensemble.ingest.adapters.llm_judge import iter_examples
from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest

app = typer.Typer(add_completion=False, help="LLM Ensemble – data ingest CLI")


def _json_dumps(obj: JudgingExample) -> str:
    return obj.model_dump_json()


@app.command("ingest")
def ingest(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset adapter to use (e.g., llm-judge)"
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
        ingest --dataset llm-judge --data-dir ./data --limit 100
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

    typer.echo(f"Run ID: {run_id}", err=True)
    typer.echo(f"Output: {output_file}", err=True)

    # Process examples
    count = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as sink:
        for ex in iter_examples(data_dir):
            sink.write(_json_dumps(ex) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    typer.echo(f"Wrote {count} examples", err=True)

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
        },
    )

    typer.echo(f"Manifest: {run_dir / 'manifest.json'}", err=True)


if __name__ == "__main__":
    app()
