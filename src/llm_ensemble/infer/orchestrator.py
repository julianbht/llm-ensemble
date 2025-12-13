"""Orchestrator for the infer CLI.

This module handles infrastructure concerns for the inference pipeline:
- Loading configurations from YAML files
- Setting up run directories and logging
- Building manifests with git info and execution parameters
- Instantiating adapters from registries
- Delegating business logic to domain service

It is separated from the CLI entry point (infer_cli.py) for testability.
"""
from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.infer_run_config import InferRunConfig
from llm_ensemble.infer.schemas.infer_run_context import IngestRunContext
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.infer.inference_service import InferenceService
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
from llm_ensemble.infer.adapters.io_factory import IOAdapterFactory
from llm_ensemble.infer.adapters.provider_factory import ProviderFactory


def run_inference(
    model_config_name: str,
    provider_name: str,
    prompt_template_name: str,
    retry_config_name: str,
    io_name: str,
    input_run_name: str,
    logging_config_name: str = "observability",
    run_name: Optional[str] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    official: bool = False,
    notes: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    """Run LLM inference on judging examples with full provenance.

    Infrastructure orchestration that coordinates:
    - Loading all YAML configurations
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters from registries
    - Running inference and writing output

    Args:
        model_config_name: Name of the model config file (e.g., "gpt-oss-20b")
        provider_name: Provider name for registry lookup (e.g., "openrouter", "ollama")
        prompt_template_name: Prompt template name (bundles builder and parser, e.g., "thomas-simple")
        retry_config_name: Name of the retry config file (e.g., "standard")
        io_name: I/O format name (e.g., "db_to_json", "db_to_db")
        input_run_name: Ingest run identifier (e.g., "my_ingest_run")
        logging_config_name: Name of the logging config file (defaults to "observability")
        run_name: Custom run ID (auto-generates if not provided)
        start_idx: Start index into NormalizedDataset (None = start from beginning)
        end_idx: End index into NormalizedDataset (None = process until end)
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        tag: Tag name for easy reference by downstream CLIs (e.g., "my-experiment")

    Raises:
        FileNotFoundError: If config or input run doesn't exist
        ValueError: If adapter is not recognized or config is invalid
    """

    # Load all configurations from YAML files
    model_config = ModelConfig.load(model_config_name)
    retry_config = RetryConfig.load(retry_config_name)
    logging_config = LoggingConfig.load(logging_config_name)

    # Create metadata-only entities (no adapter instantiation yet)
    prompt_template = PromptTemplateFactory.get_metadata(prompt_template_name)

    # Get provider adapter to extract metadata
    base_provider = ProviderFactory.create(
        provider_name=provider_name,
        model_config=model_config,
    )
    provider_entity = base_provider.get_provider()

    # Create execution context entity (CLI args for execution)
    execution_context = IngestRunContext(
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Create run config entity (bundles model, provider, template, retry configs, and execution context)
    run_config = InferRunConfig(
        model_cfg=model_config,
        provider=provider_entity,
        prompt_template=prompt_template,
        retry_config=retry_config,
        ingest_run_context=execution_context,
    )

    # Create run info entity (metadata only, uses name hints from config)
    run_info = InferRunInfo.create(
        name_hints=run_config.get_name_hints(),
        run_name=run_name,
        official=official,
        notes=notes,
    )

    # Get run directory from run_info and create it
    run_dir = run_info.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create tag file if tag provided
    if tag:
        TagManager.create_tag(run_dir, tag)

    # Set up log file path if saving logs
    log_file_path = run_dir / "run.log" if logging_config.save_logs else None

    # Initialize structlog logger with config
    logger = configure_logger(
        cli_name="infer",
        run_name=run_name,
        run_type=run_info.run_type,
        pretty_print=logging_config.pretty_print,
        save_logs=logging_config.save_logs,
        log_file_path=log_file_path,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
    )

    logger.info(
        InferLogEvent.INFER_STARTED,
        model=model_config_name,
        provider=provider_name,
        io_format=io_name,
        prompt_template=prompt_template_name,
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Instantiate adapters from config
    # Prompt template adapters from config
    prompt_builder_adapter, response_parser_adapter = run_config.prompt_template.get_adapters()

    # I/O adapters
    input_adapter = IOAdapterFactory.create_reader(io_name)
    output_adapter = IOAdapterFactory.create_writer(io_name)

    # Wrap provider with retry logic
    from llm_ensemble.infer.adapters.retrying_provider import RetryingProvider
    provider_adapter = RetryingProvider(base_provider, retry_config)

    # Create domain service by injecting adapters
    service = InferenceService(
        input_adapter=input_adapter,
        output_adapter=output_adapter,
        prompt_builder=prompt_builder_adapter,
        llm_provider=provider_adapter,
        response_parser=response_parser_adapter,
    )

    # Run inference pipeline
    try:
        summary = service.run_inference(
            run_info=run_info,
            run_config=run_config,
        )
        judgement_count = summary.judgement_count
        logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=judgement_count)

        # Write standalone summary.json for convenience (not source of truth)
        write_standalone_summary(summary, run_dir)
        logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(run_dir / "summary.json"))

    except Exception as e:
        logger.error(InferLogEvent.INFER_FAILED, error=str(e))
        raise

    logger.info(
        InferLogEvent.INFER_COMPLETE,
        total_judgements=summary.judgement_count,
        parsing_failures=summary.error_count,
        avg_latency_ms=f"{summary.avg_latency_ms:.1f}",
    )

    # Log warnings summary if any warnings were collected
    if summary.warnings_summary and sum(summary.warnings_summary.values()) > 0:
        total_warnings = sum(summary.warnings_summary.values())
        logger.info(
            InferLogEvent.WARNINGS_COLLECTED,
            total_warnings=total_warnings,
            **summary.warnings_summary
        )

    # Log where logs were saved if enabled
    if logging_config.save_logs:
        logger.info(InferLogEvent.LOGS_SAVED, path=str(run_dir / "run.log"))
