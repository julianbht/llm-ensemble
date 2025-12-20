"""Inference CLI - Driving Adapter

CLI Layer - Driving Adapter

This is the entry point and driving adapter for the inference pipeline.
In hexagonal architecture, the CLI is the driving adapter that:
1. Parses arguments
2. Calls the dependency configurator to build the application
3. Sets up CLI-specific infrastructure (run directories, logging)
4. Executes the application via its driving port (ForRunningInference)
5. Finalizes CLI-specific outputs (summaries, terminal logging)

The CLI handles all adapter concerns (I/O, infrastructure, presentation).
Tested via CLI integration tests.
"""
from __future__ import annotations
from typing import Tuple
import typer

from llm_ensemble.infer.startup.dependency_configurator import build_application
from llm_ensemble.infer.startup.adapter_config import AdapterConfig, ExecutionParams
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary
from llm_ensemble.libs.runtime.env import load_runtime_config

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

from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – inference CLI",
    pretty_exceptions_enable=False,  # Disable Rich verbose tracebacks
)


def _setup_infrastructure(
    run_config: InferRunConfig,
    execution_params: ExecutionParams,
    logging_config_name: str,
) -> Tuple[InferRunInfo, object]:
    """Setup CLI-specific infrastructure: run directory and file-based logging.

    Args:
        run_config: Domain configuration bundle
        execution_params: CLI execution parameters
        logging_config_name: Name of logging config file

    Returns:
        Tuple of (run_info, logger)
    """
    logging_config = LoggingConfig.load(logging_config_name)

    # Generate run name and create run directory
    name_hints = [
        run_config.model_cfg.name_hint,
        run_config.prompt_template.name,
        run_config.provider.name,
    ]
    run_info = InferRunInfo.create(
        name_hints=name_hints,
        run_name=execution_params.run_name,
        official=execution_params.official,
        notes=execution_params.notes,
    )

    # Create run directory (CLI-specific: file-based output)
    run_dir = run_info.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create tag symlink if requested (CLI-specific)
    if execution_params.tag:
        TagManager.create_tag(run_dir, execution_params.tag)

    # Setup file-based logging (CLI-specific)
    log_file = run_dir / "run.log" if logging_config.save_logs else None
    logger = configure_logger(
        cli_name="infer",
        run_name=execution_params.run_name,
        run_type=run_info.run_type,
        pretty_print=logging_config.pretty_print,
        save_logs=logging_config.save_logs,
        log_file_path=log_file,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
    )

    # Log startup to terminal (CLI-specific)
    logger.info(
        InferLogEvent.INFER_STARTED,
        model=run_config.model_cfg.name_hint,
        provider=run_config.provider.name,
        io_format=run_config.io_name,
        prompt_template=run_config.prompt_template.name,
        input_run_name=run_config.ingest_run_context.input_run_name,
        start_idx=run_config.ingest_run_context.start_idx,
        end_idx=run_config.ingest_run_context.end_idx,
    )

    return run_info, logger


def _finalize_run(
    summary: InferRunSummary,
    run_info: InferRunInfo,
    logger: object,
) -> None:
    """Finalize CLI outputs: write summary to file and log completion to terminal.

    Args:
        summary: Inference run summary from application
        run_info: Run metadata
        logger: Configured logger instance
    """
    # Log completion to terminal (CLI-specific)
    logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=summary.judgement_count)

    # Write summary to file (CLI-specific: file-based persistence)
    write_standalone_summary(summary, run_info.run_dir)
    logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(run_info.run_dir / "summary.json"))

    # Log final statistics to terminal (CLI-specific)
    logger.info(
        InferLogEvent.INFER_COMPLETE,
        total_judgements=summary.judgement_count,
        parsing_failures=summary.error_count,
        avg_latency_ms=f"{summary.avg_latency_ms:.1f}",
    )

    # Log warnings if any (CLI-specific)
    if summary.warnings_summary and sum(summary.warnings_summary.values()) > 0:
        total_warnings = sum(summary.warnings_summary.values())
        logger.info(
            InferLogEvent.WARNINGS_COLLECTED,
            total_warnings=total_warnings,
            **summary.warnings_summary
        )

    # Log file location (CLI-specific)
    log_file = run_info.run_dir / "run.log"
    if log_file.exists():
        logger.info(InferLogEvent.LOGS_SAVED, path=str(log_file))


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
    """Run LLM inference on judging examples.

    CLI driving adapter that:
    1. Builds configuration objects
    2. Gets application from dependency configurator
    3. Sets up CLI infrastructure (run dirs, logging)
    4. Executes application business logic
    5. Finalizes CLI outputs
    """
    # Build adapter selection config (WHICH adapters to use)
    adapter_config = AdapterConfig(
        model_config_name=model_cfg,
        provider_name=provider,
        prompt_template_name=prompt_template,
        retry_config_name=retry_cfg,
        io_name=io_cfg,
        logging_config_name=log_cfg,
    )

    # Build execution parameters (HOW to execute)
    execution_params = ExecutionParams(
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
        run_name=run_name,
        official=official,
        notes=notes,
        tag=tag,
    )

    # Build application via dependency configurator (composition root)
    application, run_config = build_application(
        adapter_config=adapter_config,
        execution_params=execution_params,
    )

    # Setup CLI-specific infrastructure (run directory, logging)
    run_info, logger = _setup_infrastructure(
        run_config=run_config,
        execution_params=execution_params,
        logging_config_name=adapter_config.logging_config_name,
    )

    # Execute application (pure business logic via ForRunningInference interface)
    summary = application.execute(
        run_info=run_info,
        run_config=run_config,
    )

    # Finalize CLI outputs (write summary, log completion to terminal)
    _finalize_run(summary, run_info, logger)

    
if __name__ == "__main__":
    app()
