from __future__ import annotations
from pathlib import Path
from typing import Optional
import sys
import json

import typer

from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.infer.adapters.config_loader import load_model_config, get_endpoint_url
from llm_ensemble.infer.adapters.huggingface import iter_judgements

app = typer.Typer(add_completion=False, help="LLM Ensemble – inference CLI")


def _read_examples(input_path: Path) -> list[JudgingExample]:
    """Read NDJSON examples from file."""
    examples = []
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                examples.append(JudgingExample(**json.loads(line)))
    return examples


def _json_dumps(judgement) -> str:
    """Serialize ModelJudgement to JSON."""
    return judgement.model_dump_json()


@app.command("infer")
def infer(
    model: str = typer.Option(
        ..., "--model", "-m", help="Model ID (e.g., phi3-mini)"
    ),
    input_file: Path = typer.Option(
        ..., "--input", "-i", exists=True, file_okay=True, readable=True,
        help="Input NDJSON file with JudgingExample records (from ingest CLI)"
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Write judgements to this file (defaults to stdout)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Process at most N examples"
    ),
    config_dir: Optional[Path] = typer.Option(
        None, "--config-dir", help="Path to model configs directory"
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt", help="Override default system prompt"
    ),
):
    """Run LLM inference on judging examples and output structured judgements.

    Reads JudgingExample records from NDJSON, runs them through the specified
    model, and outputs ModelJudgement records as NDJSON.

    12-factor friendly: reads from files, writes to stdout by default.
    Configuration via flags and environment variables (HF_TOKEN, etc.).

    Example:
        infer --model phi3-mini --input samples.ndjson --out judgements.ndjson --limit 10

    Environment variables:
        HF_TOKEN: HuggingFace API token (required for HF models)
        HF_ENDPOINT_<MODEL>_URL: Override endpoint URL for specific model
    """
    # Load model config
    try:
        model_config = load_model_config(model, config_dir)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded model config: {model_config.model_id} ({model_config.provider})", err=True)

    # Get endpoint URL
    try:
        endpoint_url = get_endpoint_url(model_config)
        typer.echo(f"Endpoint: {endpoint_url}", err=True)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Read examples
    typer.echo(f"Reading examples from {input_file}...", err=True)
    examples = _read_examples(input_file)
    typer.echo(f"Loaded {len(examples)} examples", err=True)

    if limit is not None:
        examples = examples[:limit]
        typer.echo(f"Limited to {len(examples)} examples", err=True)

    # Output stream selection
    if out is None:
        sink = sys.stdout
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        sink = out.open("w", encoding="utf-8", newline="\n")

    # Run inference
    count = 0
    try:
        typer.echo(f"Starting inference...", err=True)
        for judgement in iter_judgements(
            iter(examples),
            model_config,
            endpoint_url,
            system_prompt,
        ):
            sink.write(_json_dumps(judgement) + "\n")
            count += 1

            # Progress logging
            if count % 10 == 0:
                typer.echo(f"Processed {count}/{len(examples)} examples...", err=True)

    except Exception as e:
        typer.echo(f"Error during inference: {e}", err=True)
        raise typer.Exit(1)
    finally:
        if out is not None:
            sink.close()

    typer.echo(f"Wrote {count} judgements", err=True)


if __name__ == "__main__":
    app()
