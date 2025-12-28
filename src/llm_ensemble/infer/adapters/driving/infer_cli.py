"""Inference CLI - Driving Adapter

CLI Layer - Driving Adapter

This is a thin driving adapter that:
1. Parses CLI arguments
2. Calls the dependency configurator to build the application
3. Executes the application via its driving port (ForRunningInference)

The application handles all backend concerns (infrastructure setup, logging,
inference execution, result persistence). This adapter simply triggers it
and all logging appears in the terminal automatically.

Tested via CLI integration tests.
"""
from __future__ import annotations
import typer

from llm_ensemble.infer.startup.dependency_configurator import build_application
from llm_ensemble.infer.startup.config_loader import build_run_name_hints
from llm_ensemble.libs.runtime.run_name import generate_run_name

from llm_ensemble.libs.cli.params import (
    RunName,
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

app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – inference CLI",
    pretty_exceptions_enable=False,  # Disable verbose tracebacks
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
    official: Official = False,
    notes: Notes = None,
    tag: Tag = None,
):
    """Run LLM inference on judging examples.

    Thin CLI driving adapter that builds the application and executes it.
    All backend logic (infrastructure, logging, inference) handled by application.
    """
    # Generate run name if not given
    if run_name is None:
        name_hints = build_run_name_hints(
            model_config_name=model_cfg,
            prompt_template_name=prompt_template,
            provider_name=provider,
            io_name=io_cfg,
        )
        run_name = generate_run_name(name_hints)

    # Build application
    application = build_application(
        provider_name=provider,
        io_name=io_cfg,
        prompt_template_name=prompt_template,
        model_config_name=model_cfg,
        retry_config_name=retry_cfg,
        run_name=run_name,
        official=official,
        tag=tag,
    )

    # Run application
    application.run_inference(
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()
