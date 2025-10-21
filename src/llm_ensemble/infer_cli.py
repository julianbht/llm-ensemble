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
        ..., "--model", "-m", help="Model config name (e.g., 'gpt-oss-20b' for configs/models/gpt-oss-20b.yaml)"
    ),
    input_file: Path = typer.Option(
        ..., "--input", "-i", exists=True, file_okay=True, readable=True,
        help="Input file with JudgingExample records (from ingest CLI)"
    ),
    io_format: str = typer.Option(
        "ndjson", "--io", help="I/O format config name (e.g., 'ndjson' for configs/io/ndjson.yaml)"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Custom run ID (auto-generates if not provided)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Process at most N examples"
    ),
    config_dir: Optional[Path] = typer.Option(
        None, "--config-dir", help="Path to model configs directory (defaults to configs/models)"
    ),
    io_dir: Optional[Path] = typer.Option(
        None, "--io-dir", help="Path to I/O configs directory (defaults to configs/io)"
    ),
    prompts_dir: Optional[Path] = typer.Option(
        None, "--prompts-dir", help="Path to prompts directory (defaults to configs/prompts)"
    ),
    prompt: str = typer.Option(
        "thomas-et-al-prompt", "--prompt", "-p", help="Prompt config name (located in ./configs/prompts)"
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

    Reads JudgingExample records, runs inference, and writes ModelJudgement records
    to artifacts/runs/<run_id>/judgements.<format> with manifest.

    All behavior is explicitly configured via config files - no implicit defaults.

    Example:
        infer --model gpt-oss-20b --input artifacts/runs/20250115_143022_llm-judge/samples.ndjson

        # Explicit I/O format (same as default)
        infer --model gpt-oss-20b --input data.ndjson --io ndjson

    Environment variables:
        OPENROUTER_API_KEY: OpenRouter API key (required for OpenRouter models)
        HF_TOKEN: HuggingFace API token (required for HF models)
    """
    try:
        run_inference(
            model=model,
            input_file=input_file,
            io_format=io_format,
            run_id=run_id,
            limit=limit,
            config_dir=config_dir,
            io_dir=io_dir,
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
