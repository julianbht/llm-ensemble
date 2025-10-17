from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

from llm_ensemble.infer.orchestrator import run_inference
from llm_ensemble.libs.runtime.env import load_runtime_config

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – inference CLI")


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
    try:
        run_inference(
            model=model,
            input_file=input_file,
            run_id=run_id,
            limit=limit,
            config_dir=config_dir,
            prompts_dir=prompts_dir,
            prompt=prompt,
            save_logs=save_logs,
            official=official,
            notes=notes,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
