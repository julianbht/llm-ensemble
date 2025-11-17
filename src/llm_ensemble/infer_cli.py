from __future__ import annotations
from typing import Annotated
import typer

from llm_ensemble.infer.orchestrator import run_inference
from llm_ensemble.infer.config_loaders import load_model_config, load_prompt_config, load_retry_config
from llm_ensemble.libs.config import load_io_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides
from llm_ensemble.libs.cli.params import (
    RunName,
    LogCfg,
    Official,
    Notes,
    Override,
    Limit,
    ModelCfg,
    PromptCfg,
    InferIoCfg,
    RetryCfg,
    Tag,
    IngestRunInput,
)

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – inference CLI",
    pretty_exceptions_enable=False,  # Disable Rich verbose tracebacks
)


@app.command("infer")
def infer(
    # Required parameters with validation
    model_cfg: ModelCfg,
    prompt_cfg: PromptCfg,
    io_cfg: InferIoCfg,
    input_run_name: IngestRunInput,
    # Optional parameters
    retry_cfg: RetryCfg = "standard",
    limit: Limit = None,
    run_name: RunName = None,
    log_cfg: LogCfg = "observability",
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
    tag: Tag = None,
):
    """Run LLM inference on judging samples and output structured judgements.
        OPENROUTER_API_KEY: OpenRouter API key (required for OpenRouter models)
        HF_TOKEN: HuggingFace API token (required for HF models)
    """
    # Resolve tag if input starts with @ (already validated by RunInputParamType)
    from llm_ensemble.libs.runtime.tag_manager import TagManager
    input_run_name = TagManager.resolve_input(input_run_name, "ingest")
    
    # Load configurations
    model_config = load_model_config(model_cfg)
    prompt_config = load_prompt_config(prompt_cfg)
    retry_config = load_retry_config(retry_cfg)
    io_config = load_io_config(io_cfg, cli_name="infer")
    logging_config = load_logging_config(log_cfg or "observability")

    # Parse and route overrides if provided
    if override:
        overrides = parse_and_route_overrides(override)

        # Apply routed overrides to each config
        if overrides['model']:
            model_config = apply_overrides(model_config, overrides['model'])
        if overrides['prompt']:
            prompt_config = apply_overrides(prompt_config, overrides['prompt'])
        if overrides['io']:
            io_config = apply_overrides(io_config, overrides['io'])

    # Run inference with final configs
    run_inference(
        model_config=model_config,
        prompt_config=prompt_config,
        retry_config=retry_config,
        io_config=io_config,
        logging_config=logging_config,
        input_run_name=input_run_name,
        model_config_name=model_cfg,
        prompt_config_name=prompt_cfg,
        retry_config_name=retry_cfg,
        io_config_name=io_cfg,
        run_name=run_name,
        limit=limit,
        official=official,
        notes=notes,
        tag=tag,
    )

    
if __name__ == "__main__":
    app()
