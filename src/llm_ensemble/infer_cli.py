from __future__ import annotations
import typer

from llm_ensemble.infer.application.orchestrator import run_inference
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.runtime.tag_manager import TagManager

from llm_ensemble.libs.cli.params import (
    RunName,
    LogCfg,
    Official,
    Notes,
    StartIdx,
    EndIdx,
    ModelCfg,
    PromptTemplate,
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
    # Required parameters
    model_cfg: ModelCfg,
    provider: Provider,
    prompt_template: PromptTemplate,
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
    tag: Tag = None,
):
    """Run LLM inference."""
    # Resolve tag if input starts with @ (already validated by RunInputParamType)
    input_run_name = TagManager.resolve_input(input_run_name, "ingest")

    # Delegate to orchestrator (config loading happens there)
    run_inference(
        model_config_name=model_cfg,
        provider_name=provider,
        prompt_template_name=prompt_template,
        retry_config_name=retry_cfg,
        io_name=io_cfg,
        logging_config_name=log_cfg,
        input_run_name=input_run_name,
        run_name=run_name,
        start_idx=start_idx,
        end_idx=end_idx,
        official=official,
        notes=notes,
        tag=tag,
    )

    
if __name__ == "__main__":
    app()
