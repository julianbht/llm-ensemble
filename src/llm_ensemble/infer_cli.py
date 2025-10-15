from __future__ import annotations
from pathlib import Path
from typing import Optional
import sys
import json

import typer

from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.infer.adapters.config_loader import load_model_config
from llm_ensemble.infer.adapters.inference_router import iter_judgements
from llm_ensemble.libs.logging import configure_logging
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.runtime.git_utils import get_git_info

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
        ..., "--model", "-m", help="Model ID (e.g., gpt-oss-20b)"
    ),
    input_file: Path = typer.Option(
        ..., "--input", "-i", exists=True, file_okay=True, readable=True,
        help="Input NDJSON file with JudgingExample records (from ingest CLI)"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Custom run ID (auto-generates if not provided)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Process at most N examples"
    ),
    config_dir: Optional[Path] = typer.Option(
        None, "--config-dir", help="Path to model configs directory"
    ),
    prompts_dir: Optional[Path] = typer.Option(
        None, "--prompts-dir", help="Path to prompt templates directory (defaults to configs/prompts)"
    ),
    prompt: str = typer.Option(
        "thomas-et-al-prompt", "--prompt", "-p", help="Prompt template name (without .jinja extension)"
    ),
):
    """Run LLM inference on judging examples and output structured judgements.

    Reads JudgingExample records from NDJSON, runs inference, and writes
    ModelJudgement records to artifacts/runs/<run_id>/judgements.ndjson with manifest.

    Example:
        infer --model gpt-oss-20b --input artifacts/runs/20250115_143022_llm-judge/samples.ndjson

    Environment variables:
        OPENROUTER_API_KEY: OpenRouter API key (required for OpenRouter models)
        HF_TOKEN: HuggingFace API token (required for HF models)
    """
    # Load model config
    try:
        model_config = load_model_config(model, config_dir)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded model config: {model_config.model_id} ({model_config.provider})", err=True)

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(model_config.model_id)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="infer")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "judgements.ndjson"
    log_file = run_dir / "logs.jsonl"

    # Configure structured logging
    git_info = get_git_info()
    logger = configure_logging(
        cli_name="infer",
        run_id=run_id,
        log_file=log_file,
        git_sha=git_info.get("git_sha"),
    )

    typer.echo(f"Run ID: {run_id}", err=True)
    typer.echo(f"Output: {output_file}", err=True)

    # Read examples
    typer.echo(f"Reading examples from {input_file}...", err=True)
    examples = _read_examples(input_file)
    typer.echo(f"Loaded {len(examples)} examples", err=True)

    if limit is not None:
        examples = examples[:limit]
        typer.echo(f"Limited to {len(examples)} examples", err=True)

    logger.info(
        "inference_started",
        model=model_config.model_id,
        provider=model_config.provider,
        num_samples=len(examples),
        input_file=str(input_file),
        prompt_template=prompt,
    )

    # Run inference
    count = 0
    error_count = 0
    total_latency_ms = 0.0

    try:
        typer.echo(f"Starting inference with prompt: {prompt}", err=True)
        with output_file.open("w", encoding="utf-8", newline="\n") as sink:
            for judgement in iter_judgements(
                iter(examples),
                model_config,
                prompts_dir=prompts_dir,
                prompt_template_name=prompt,
            ):
                sink.write(_json_dumps(judgement) + "\n")
                count += 1
                total_latency_ms += judgement.latency_ms

                # Track errors
                if judgement.label is None:
                    error_count += 1
                    logger.debug(
                        "inference_failed",
                        query_id=judgement.query_id,
                        doc_id=judgement.doc_id,
                        warnings=judgement.warnings,
                    )
                else:
                    logger.debug(
                        "inference_success",
                        query_id=judgement.query_id,
                        doc_id=judgement.doc_id,
                        label=judgement.label,
                        latency_ms=judgement.latency_ms,
                    )

                # Progress logging
                if count % 10 == 0:
                    typer.echo(f"Processed {count}/{len(examples)} examples...", err=True)

        logger.info(
            "inference_completed",
            judgement_count=count,
            error_count=error_count,
            avg_latency_ms=total_latency_ms / count if count > 0 else 0,
        )

    except Exception as e:
        logger.error("inference_failed", error=str(e), error_type=type(e).__name__)
        typer.echo(f"Error during inference: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Wrote {count} judgements ({error_count} errors)", err=True)

    # Write manifest
    write_manifest(
        run_dir=run_dir,
        cli_name="infer",
        cli_args={
            "model": model,
            "input_file": str(input_file),
            "limit": limit,
            "prompt": prompt,
        },
        metadata={
            "model_config": model_config.model_dump(),
            "prompt_template": prompt,
            "judgement_count": count,
            "error_count": error_count,
            "avg_latency_ms": total_latency_ms / count if count > 0 else 0,
            "output_file": str(output_file),
            "log_file": str(log_file),
        },
    )

    typer.echo(f"Manifest: {run_dir / 'manifest.json'}", err=True)


if __name__ == "__main__":
    app()
