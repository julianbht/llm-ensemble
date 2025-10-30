from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

from llm_ensemble.infer.orchestrator import run_inference
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.utils.config_overrides import parse_overrides
from llm_ensemble.libs.cli.common_params import IoFormat, RunId, SaveLogs, Official, Notes, Override

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – inference CLI")


@app.command("infer")
def infer(
    # Required parameters
    io_format: IoFormat,
    model: str = typer.Option(
        ..., "--model", "-m", help="Model config name (e.g., 'gpt-oss-20b' for configs/models/gpt-oss-20b.yaml)"
    ),
    input_file: Path = typer.Option(
        ..., "--input", "-i", exists=True, file_okay=True, readable=True,
        help="Input file with JudgingExample records (from ingest CLI)"
    ),
    # Optional parameters
    prompt: str = typer.Option(
        "thomas-et-al-prompt", "--prompt", "-p", help="Prompt config name (located in ./configs/prompts)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Process at most N examples"
    ),
    run_id: RunId = None,
    save_logs: SaveLogs = False,
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
):
    """Run LLM inference on judging examples and output structured judgements.

    Reads JudgingExample records, runs inference, and writes ModelJudgement records
    to artifacts/runs/<run_id>/judgements.<format> with manifest.

    All behavior is explicitly configured via config files - no implicit defaults.

    Examples:
        # Basic usage
        infer --model gpt-oss-20b --input data.ndjson

        # Override model parameters
        infer --model gpt-oss-20b --input data.ndjson \\
              --override default_params.temperature=0.7 \\
              --override default_params.max_tokens=512

        # Override prompt variables
        infer --model gpt-oss-20b --input data.ndjson \\
              --prompt thomas-et-al-prompt \\
              --override variables.role=false

    Override format:
        Model params:    --override default_params.temperature=0.7
        Prompt vars:     --override variables.role=false
        I/O adapters:    --override reader=custom_reader

        See config files in configs/ for available fields.
        Overrides are tracked in manifest for reproducibility.

    Environment variables:
        OPENROUTER_API_KEY: OpenRouter API key (required for OpenRouter models)
        HF_TOKEN: HuggingFace API token (required for HF models)
    """
    try:
        # Parse overrides
        config_overrides = parse_overrides(override) if override else {}

        run_inference(
            model=model,
            input_file=input_file,
            prompt=prompt,
            io_format=io_format,
            run_id=run_id,
            limit=limit,
            save_logs=save_logs,
            official=official,
            notes=notes,
            config_overrides=config_overrides,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
