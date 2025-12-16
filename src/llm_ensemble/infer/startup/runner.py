"""Inference pipeline runner.

Startup Layer - Infrastructure Orchestration (BlueZoneRunner equivalent)

Clean phased execution:
1. Build InferRunConfig (THE canonical config - serializable domain entity)
2. Setup infrastructure (run directory, logging)
3. Build application (lookup adapters, assemble hexagon)
4. Execute use case
5. Finalize (write summary, final logs)

This is the composition root - NOT unit tested.
Tested via CLI integration tests.
"""
from __future__ import annotations
from typing import Tuple
from logging import Logger

from llm_ensemble.infer.startup.adapter_config import AdapterConfig, ExecutionParams
from llm_ensemble.infer.application.inference_use_case import InferenceUseCase
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary

from llm_ensemble.infer.adapters.io_factory import IOAdapterFactory
from llm_ensemble.infer.adapters.provider_factory import ProviderFactory
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
from llm_ensemble.infer.adapters.retrying_provider import RetryingProvider

from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager


def run_inference(
    adapter_config: AdapterConfig,
    execution_params: ExecutionParams,
) -> None:
    """Run inference pipeline (composition root)."""
    run_config = _build_run_config(adapter_config, execution_params)
    run_info, logger = _setup_infrastructure(run_config, execution_params, adapter_config.logging_config_name)
    use_case = _build_application(run_config)
    summary = use_case.execute(run_info=run_info, run_config=run_config)
    _finalize_run(summary, run_info, logger)

def  _build_run_config(
    adapter_config: AdapterConfig,
    execution_params: ExecutionParams,
) -> InferRunConfig:
    """Load all configs and build InferRunConfig directly."""
    input_run_name = TagManager.resolve_input(execution_params.input_run_name, "ingest")
    
    return InferRunConfig(
        model_cfg=ModelConfig.load(adapter_config.model_config_name),
        retry_config=RetryConfig.load(adapter_config.retry_config_name),
        prompt_template=PromptTemplateFactory.create(adapter_config.prompt_template_name),
        provider=Provider(name=adapter_config.provider_name),
        io_name=adapter_config.io_name,
        ingest_run_context=IngestRunContext(
            input_run_name=input_run_name,
            start_idx=execution_params.start_idx,
            end_idx=execution_params.end_idx,
        ),
    )


def _setup_infrastructure(
    run_config: InferRunConfig,
    execution_params: ExecutionParams,
    logging_config_name: str,
) -> Tuple[InferRunInfo, Logger]:
    """Setup run directory, logging. Returns run_info and logger."""
    logging_config = LoggingConfig.load(logging_config_name)
    
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
    
    run_dir = run_info.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if execution_params.tag:
        TagManager.create_tag(run_dir, execution_params.tag)
    
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


def _build_application(run_config: InferRunConfig) -> InferenceUseCase:
    """Instantiate adapters and build application (hexagon)."""
    input_port = IOAdapterFactory.create_reader(run_config.io_name)
    output_port = IOAdapterFactory.create_writer(run_config.io_name)
    
    template_adapter = PromptTemplateFactory.get_adapter_class(run_config.prompt_template.name)()
    prompt_builder = template_adapter.get_builder()
    response_parser = template_adapter.get_parser()
    
    base_provider = ProviderFactory.create(run_config.provider.name, run_config.model_cfg)
    llm_provider = RetryingProvider(base_provider, run_config.retry_config)
    
    return InferenceUseCase(
        input_port=input_port,
        output_port=output_port,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        response_parser=response_parser,
    )


def _finalize_run(
    summary: InferRunSummary,
    run_info: InferRunInfo,
    logger: Logger,
) -> None:
    """Write summary and final logs."""
    logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=summary.judgement_count)
    
    write_standalone_summary(summary, run_info.run_dir)
    logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(run_info.run_dir / "summary.json"))
    
    logger.info(
        InferLogEvent.INFER_COMPLETE,
        total_judgements=summary.judgement_count,
        parsing_failures=summary.error_count,
        avg_latency_ms=f"{summary.avg_latency_ms:.1f}",
    )
    
    if summary.warnings_summary and sum(summary.warnings_summary.values()) > 0:
        total_warnings = sum(summary.warnings_summary.values())
        logger.info(
            InferLogEvent.WARNINGS_COLLECTED,
            total_warnings=total_warnings,
            **summary.warnings_summary
        )
    
    log_file = run_info.run_dir / "run.log"
    if log_file.exists():
        logger.info(InferLogEvent.LOGS_SAVED, path=str(log_file))
