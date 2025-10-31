from __future__ import annotations
import typer

from llm_ensemble.infer.orchestrator import run_inference
from llm_ensemble.infer.config_loaders import load_model_config, load_prompt_config
from llm_ensemble.libs.config import load_io_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides
from llm_ensemble.libs.cli.common_params import InputPath, IoCfg, RunId, LogCfg, Official, Notes, Override, Limit

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – inference CLI")


@app.command("infer")
def infer(
    # Required parameters
    input_path: InputPath,
    io_cfg: IoCfg,
    model_cfg: str = typer.Option(
        ...,
        "--model-cfg",
        help=f"Model config name. Configs in {PathManager.get_model_configs_dir().relative_to(PathManager.get_project_root())}"
    ),
    # Required parameters (continued)
    prompt_cfg: str = typer.Option(
        ...,
        "--prompt-cfg",
        help=f"Prompt config name. Configs in {PathManager.get_prompts_dir().relative_to(PathManager.get_project_root())}"
    ),
    # Optional parameters
    limit: Limit = None,
    run_id: RunId = None,
    log_cfg: LogCfg = None,
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
):
    """Run LLM inference on judging examples and output structured judgements.

    Reads JudgingExample records, runs inference, and writes ModelJudgement records
    with full provenance metadata.

    All behavior is explicitly configured via config files - no implicit defaults.

    Examples:
        # Basic usage
        infer --model-cfg gpt-oss-20b --prompt-cfg thomas-et-al-prompt --input data.ndjson --io-cfg ndjson

        # With JSON logging and no file saving
        infer --model-cfg gpt-oss-20b --prompt-cfg thomas-et-al-prompt --input data.ndjson --io-cfg ndjson --log-cfg json

        # With console-only logging (no file saving)
        infer --model-cfg gpt-oss-20b --prompt-cfg thomas-et-al-prompt --input data.ndjson --io-cfg ndjson --log-cfg console-only

        # Override model parameters (note: prefix-based routing)
        infer --model-cfg gpt-oss-20b --prompt-cfg thomas-et-al-prompt --input data.ndjson --io-cfg ndjson \\
              --override model.default_params.temperature=0.7 \\
              --override model.default_params.max_tokens=512

        # Override prompt variables
        infer --model-cfg gpt-oss-20b --prompt-cfg thomas-et-al-prompt --input data.ndjson --io-cfg ndjson \\
              --override prompt.variables.role=false

        # Override I/O adapters
        infer --model-cfg gpt-oss-20b --prompt-cfg thomas-et-al-prompt --input data.ndjson --io-cfg ndjson \\
              --override io.reader=custom_reader

    Override format (prefix-based routing):
        Model params:    --override model.default_params.temperature=0.7
        Prompt vars:     --override prompt.variables.role=false
        I/O adapters:    --override io.reader=custom_reader

        Prefix must be one of: model, prompt, io

    Environment variables:
        OPENROUTER_API_KEY: OpenRouter API key (required for OpenRouter models)
        HF_TOKEN: HuggingFace API token (required for HF models)
    """
    # Load configurations
    model_config = load_model_config(model_cfg)
    prompt_config = load_prompt_config(prompt_cfg)
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
        io_config=io_config,
        logging_config=logging_config,
        input_file=input_path,
        model_config_name=model_cfg,
        prompt_config_name=prompt_cfg,
        io_config_name=io_cfg,
        run_id=run_id,
        limit=limit,
        official=official,
        notes=notes,
    )

    
if __name__ == "__main__":
    app()
