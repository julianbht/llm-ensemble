from __future__ import annotations
import typer

from llm_ensemble.infer.orchestrator import run_inference
from llm_ensemble.infer.config_loaders import load_model_config, load_retry_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides

from llm_ensemble.libs.cli.params import (
    RunName,
    LogCfg,
    Official,
    Notes,
    Override,
    StartIdx,
    EndIdx,
    ModelCfg,
    Prompt,
    Parser,
    Provider,
    InferIoCfg,
    RetryCfg,
    Tag,
    InferIngestRunInput,
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
    provider: Provider,
    prompt: Prompt,
    parser: Parser,
    io_cfg: InferIoCfg,
    input_run_name: InferIngestRunInput,
    # Optional parameters
    retry_cfg: RetryCfg = "standard",
    start_idx: StartIdx = None,
    end_idx: EndIdx = None,
    run_name: RunName = None,
    log_cfg: LogCfg = "observability",
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
    tag: Tag = None,
):
    """Run LLM inference."""
    # Resolve tag if input starts with @ (already validated by RunInputParamType)
    input_run_name = TagManager.resolve_input(input_run_name, "ingest")

    # Load configurations
    model_config = load_model_config(model_cfg)
    retry_config = load_retry_config(retry_cfg)
    logging_config = load_logging_config(log_cfg)

    # Parse and route overrides if provided
    if override:
        overrides = parse_and_route_overrides(override)
        if overrides['model']:
            model_config = apply_overrides(model_config, overrides['model'])

    # Run inference with registry-based adapter selection
    run_inference(
        model_config=model_config,
        provider_name=provider,
        prompt_name=prompt,
        parser_name=parser,
        retry_config=retry_config,
        io_name=io_cfg,
        logging_config=logging_config,
        input_run_name=input_run_name,
        model_config_name=model_cfg,
        retry_config_name=retry_cfg,
        run_name=run_name,
        start_idx=start_idx,
        end_idx=end_idx,
        official=official,
        notes=notes,
        tag=tag,
    )

    
if __name__ == "__main__":
    app()
