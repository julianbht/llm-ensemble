"""Inference pipeline runner.

Startup Layer - Infrastructure Orchestration (BlueZoneRunner equivalent)

Clean phased execution:
1. Load configurations
2. Setup infrastructure (run directory, logging, domain entities)
3. Build application (lookup adapters, assemble hexagon)
4. Execute use case
5. Finalize (write summary, final logs)

This is the composition root - NOT unit tested.
Tested via CLI integration tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from logging import Logger

from llm_ensemble.infer.startup.adapter_config import AdapterConfig, ExecutionParams
from llm_ensemble.infer.application.inference_use_case import InferenceUseCase
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.schemas.infer_run_summary_schema import InferRunSummary

from llm_ensemble.infer.adapters.io_factory import IOAdapterFactory
from llm_ensemble.infer.adapters.provider_factory import ProviderFactory
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
from llm_ensemble.infer.adapters.retrying_provider import RetryingProvider

from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager


@dataclass
class LoadedConfigs:
    """All loaded configurations bundled together."""
    model: ModelConfig
    retry: RetryConfig
    prompt_template: PromptTemplate
    logging: LoggingConfig
    provider_name: str
    io_name: str
    input_run_name: str
    start_idx: Optional[int]
    end_idx: Optional[int]


@dataclass
class RunContext:
    """Runtime context for this inference run."""
    run_info: InferRunInfo
    run_config: InferRunConfig
    run_dir: Path
    logger: Logger


# ============================================================================
# Main Entry Point
# ============================================================================

def run_inference(
    adapter_config: AdapterConfig,
    execution_params: ExecutionParams,
) -> None:
    """Run inference pipeline (composition root).

    Args:
        adapter_config: Adapter selection (which adapters to use)
        execution_params: Execution parameters (how to run)

    Raises:
        FileNotFoundError: If config or input run doesn't exist
        ValueError: If adapter is not recognized or config is invalid
    """
    # Load configurations
    configs = _load_configs(adapter_config, execution_params)
    
    # Setup infrastructure
    run_context = _setup_infrastructure(configs, execution_params)
    
    # Build application
    use_case = _build_application(configs)
    
    # Execute
    summary = use_case.execute(
        run_info=run_context.run_info,
        run_config=run_context.run_config,
    )
    
    # Finalize
    _finalize_run(summary, run_context, configs)


def _load_configs(
    adapter_config: AdapterConfig,
    execution_params: ExecutionParams,
) -> LoadedConfigs:
    """Load all configurations from YAML files."""
    input_run_name = TagManager.resolve_input(execution_params.input_run_name, "ingest")
    
    return LoadedConfigs(
        model=ModelConfig.load(adapter_config.model_config_name),
        retry=RetryConfig.load(adapter_config.retry_config_name),
        prompt_template=PromptTemplateFactory.create(adapter_config.prompt_template_name),
        logging=LoggingConfig.load(adapter_config.logging_config_name),
        provider_name=adapter_config.provider_name,
        io_name=adapter_config.io_name,
        input_run_name=input_run_name,
        start_idx=execution_params.start_idx,
        end_idx=execution_params.end_idx,
    )


def _setup_infrastructure(
    configs: LoadedConfigs,
    execution_params: ExecutionParams,
) -> RunContext:
    """Setup run directory, logging, and build domain entities."""
    
    # Build domain entities
    run_config = InferRunConfig(
        model_cfg=configs.model,
        provider=Provider(name=configs.provider_name),
        prompt_template=configs.prompt_template,
        retry_config=configs.retry,
        ingest_run_context=IngestRunContext(
            input_run_name=configs.input_run_name,
            start_idx=configs.start_idx,
            end_idx=configs.end_idx,
        ),
    )
    
    run_info = InferRunInfo.create(
        name_hints=run_config.get_name_hints(),
        run_name=execution_params.run_name,
        official=execution_params.official,
        notes=execution_params.notes,
    )
    
    # Setup directories
    run_dir = run_info.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if execution_params.tag:
        TagManager.create_tag(run_dir, execution_params.tag)
    
    # Setup logger
    log_file = run_dir / "run.log" if configs.logging.save_logs else None
    logger = configure_logger(
        cli_name="infer",
        run_name=execution_params.run_name,
        run_type=run_info.run_type,
        pretty_print=configs.logging.pretty_print,
        save_logs=configs.logging.save_logs,
        log_file_path=log_file,
        console_level=configs.logging.console_level,
        file_level=configs.logging.file_level,
    )
    
    logger.info(
        InferLogEvent.INFER_STARTED,
        model=configs.model.name_hint,
        provider=configs.provider_name,
        io_format=configs.io_name,
        prompt_template=configs.prompt_template.name,
        input_run_name=configs.input_run_name,
        start_idx=configs.start_idx,
        end_idx=configs.end_idx,
    )
    
    return RunContext(run_info, run_config, run_dir, logger)


def _build_application(configs: LoadedConfigs) -> InferenceUseCase:
    """Instantiate adapters and build application (hexagon)."""
    
    # Lookup I/O adapters
    input_port = IOAdapterFactory.create_reader(configs.io_name)
    output_port = IOAdapterFactory.create_writer(configs.io_name)
    
    # Lookup template adapters (builder + parser pair)
    template_adapter = PromptTemplateFactory.get_adapter_class(configs.prompt_template.name)()
    prompt_builder = template_adapter.get_builder()
    response_parser = template_adapter.get_parser()
    
    # Lookup provider adapter (with retry wrapper)
    base_provider = ProviderFactory.create(configs.provider_name, configs.model)
    llm_provider = RetryingProvider(base_provider, configs.retry)
    
    # Assemble hexagon
    return InferenceUseCase(
        input_port=input_port,
        output_port=output_port,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        response_parser=response_parser,
    )


def _finalize_run(
    summary: InferRunSummary,
    run_ctx: RunContext,
    configs: LoadedConfigs,
) -> None:
    """Write summary and final logs."""
    run_ctx.logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=summary.judgement_count)
    
    write_standalone_summary(summary, run_ctx.run_dir)
    run_ctx.logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(run_ctx.run_dir / "summary.json"))
    
    run_ctx.logger.info(
        InferLogEvent.INFER_COMPLETE,
        total_judgements=summary.judgement_count,
        parsing_failures=summary.error_count,
        avg_latency_ms=f"{summary.avg_latency_ms:.1f}",
    )
    
    if summary.warnings_summary and sum(summary.warnings_summary.values()) > 0:
        total_warnings = sum(summary.warnings_summary.values())
        run_ctx.logger.info(
            InferLogEvent.WARNINGS_COLLECTED,
            total_warnings=total_warnings,
            **summary.warnings_summary
        )
    
    if configs.logging.save_logs:
        run_ctx.logger.info(InferLogEvent.LOGS_SAVED, path=str(run_ctx.run_dir / "run.log"))
