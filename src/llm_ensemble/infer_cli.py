from __future__ import annotations
from pathlib import Path
from typing import Optional
import json

import typer

from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.infer.adapters.config_loader import load_model_config
from llm_ensemble.infer.adapters.inference_router import iter_judgements
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.logging.logger import get_logger

# Load runtime configuration early
load_runtime_config()

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
        ..., "--model", "-m", help="Model ID for .yaml config file(e.g., gpt-oss-20b)"
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
        # Use basic error output before logger is initialized
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(model_config.model_id)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="infer", official=official)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "judgements.ndjson"

    # Set up log file if requested
    log_file_handle = None
    if save_logs:
        log_file_path = run_dir / "run.log"
        log_file_handle = open(log_file_path, "w", encoding="utf-8")

    # Initialize logger
    logger = get_logger("infer", run_id=run_id, log_file=log_file_handle)

    logger.info("Starting inference", model=model_config.model_id, provider=model_config.provider, prompt=prompt)
    logger.info("Run directory", path=str(run_dir))
    logger.info("Output file", path=str(output_file))

    # Read examples
    logger.info("Reading examples", input_file=str(input_file))
    examples = _read_examples(input_file)
    logger.info("Loaded examples", count=len(examples))

    if limit is not None:
        examples = examples[:limit]
        logger.info("Limited examples", count=len(examples))

    # Run inference
    count = 0
    error_count = 0
    total_latency_ms = 0.0

    try:
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
                    logger.warning(
                        "Judgement error",
                        count=count,
                        query_id=judgement.query_id,
                        docid=judgement.docid,
                        warnings=judgement.warnings,
                    )
                else:
                    logger.info(
                        "Processed judgement",
                        count=count,
                        query_id=judgement.query_id,
                        docid=judgement.docid,
                        label=judgement.label,
                        latency_ms=f"{judgement.latency_ms:.1f}",
                    )

    except Exception as e:
        logger.error("Inference failed", error=str(e))
        raise typer.Exit(1)

    logger.info("Inference complete", total_judgements=count, errors=error_count, avg_latency_ms=f"{total_latency_ms / count if count > 0 else 0:.1f}")

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
