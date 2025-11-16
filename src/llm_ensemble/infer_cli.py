from __future__ import annotations
from typing import Annotated, Optional
import typer

from llm_ensemble.infer.orchestrator import run_inference
from llm_ensemble.infer.config_loaders import load_model_config, load_prompt_config, load_retry_config
from llm_ensemble.libs.config import load_io_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides
from llm_ensemble.libs.cli.common_params import (
    InputPath, RunName, LogCfg, Official, Notes, Override, Limit,
    ModelCfg, PromptCfg,
)
from llm_ensemble.libs.cli.param_types import IOConfigParamType

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
    io_cfg: Annotated[
        str,
        typer.Option(
            ...,
            "--io-cfg",
            click_type=IOConfigParamType("infer"),
            help=f"I/O config name. Configs in {(PathManager.get_configs_dir() / 'io' / 'infer').relative_to(PathManager.get_project_root())}"
        )
    ],
    input_path: InputPath = None,
    # Optional parameters
    retry_cfg: str = typer.Option(
        "standard",
        "--retry-cfg",
        help=f"Retry config name. Configs in {PathManager.get_retries_dir().relative_to(PathManager.get_project_root())}"
    ),
    limit: Limit = None,
    run_name: RunName = None,
    log_cfg: LogCfg = None,
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
):
    """Run LLM inference on judging samples and output structured judgements.
        OPENROUTER_API_KEY: OpenRouter API key (required for OpenRouter models)
        HF_TOKEN: HuggingFace API token (required for HF models)
    """
    # Load configurations
    model_config = load_model_config(model_cfg)
    prompt_config = load_prompt_config(prompt_cfg)
    retry_config = load_retry_config(retry_cfg)
    io_config = load_io_config(io_cfg, cli_name="infer")
    logging_config = load_logging_config(log_cfg or "default")

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
        input_file=input_path,
        model_config_name=model_cfg,
        prompt_config_name=prompt_cfg,
        retry_config_name=retry_cfg,
        io_config_name=io_cfg,
        run_name=run_name,
        limit=limit,
        official=official,
        notes=notes,
    )

    
if __name__ == "__main__":
    app()
