"""Orchestrator for the infer CLI.

This module handles infrastructure concerns for the inference pipeline:
- Loading configurations
- Setting up run directories and logging
- Building manifests with git info and execution parameters
- Instantiating adapters from builders
- Delegating business logic to domain service

It is separated from the CLI entry point (infer_cli.py) for testability.
"""
from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.libs.schemas import LoggingConfig
from llm_ensemble.infer.domain import InferenceService
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.runtime.git_utils import get_git_info
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.infer.prompt_builder import PromptAdapterBuilder
from llm_ensemble.infer.parser_builder import ParserAdapterBuilder
from llm_ensemble.infer.io_builder import IOAdapterBuilder
from llm_ensemble.infer.provider_builder import ProviderAdapterBuilder


def run_inference(
    model_config: ModelConfig,
    provider_name: str,
    prompt_name: str,
    parser_name: str,
    retry_config: RetryConfig,
    io_name: str,
    logging_config: LoggingConfig,
    input_run_name: str,
    model_config_name: str,
    retry_config_name: str,
    run_name: Optional[str] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    official: bool = False,
    notes: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    """Run LLM inference on judging examples with full provenance.

    Infrastructure orchestration that coordinates:
    - Reading NormalizedDataset and computing explicit indices
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters from registries
    - Running inference, attaching manifest metadata to each judgement, and writing output

    Args:
        model_config: Model configuration
        provider_name: Provider name for registry lookup (e.g., "openrouter", "ollama")
        prompt_name: Prompt name for registry lookup (e.g., "thomas-simple")
        parser_name: Parser name for registry lookup (e.g., "thomas-simple")
        retry_config: Retry configuration
        io_name: I/O format name (e.g., "db_to_json", "db_to_db")
        logging_config: Logging configuration (controls pretty printing and log saving)
        input_run_name: Ingest run identifier (e.g., "my_ingest_run")
        model_config_name: Name of the model config file (e.g., "gpt-oss-20b")
        retry_config_name: Name of the retry config file (e.g., "standard")
        run_name: Custom run ID (auto-generates if not provided)
        start_idx: Start index into NormalizedDataset (default: 0)
        end_idx: End index into NormalizedDataset (None = full dataset)
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        tag: Tag name for easy reference by downstream CLIs (e.g., "my-experiment")

    Raises:
        FileNotFoundError: If input run doesn't exist (raised by adapter during read)
        ValueError: If adapter is not recognized or config is invalid
    """

    # Generate or use provided run_name
    if run_name is None:
        run_name = generate_run_name([
            model_config.name_hint,
            prompt_name,
            parser_name,
            io_name,
        ])

    # Get git info for reproducibility
    git_info = get_git_info()

    # Create immutable run info (runtime context known before run starts)
    run_info = InferRunInfo(
        run_name=run_name,
        model_config_name=model_config_name,
        prompt_name=prompt_name,
        parser_name=parser_name,
        retry_config_name=retry_config_name,
        io_name=io_name,
        model_cfg=model_config,
        retry_config=retry_config,
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
        run_type=RunType.OFFICIAL if official else RunType.TEST,
        notes=notes,
        git_sha=git_info["git_sha"],
        git_clean=git_info["git_clean"],
        git_branch=git_info["git_branch"],
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
        prompt=prompt_name,
        parser=parser_name,
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Instantiate adapters with factories
    prompt_builder = PromptAdapterBuilder.build(prompt_name)
    response_parser = ParserAdapterBuilder.build(parser_name)
    reader = IOAdapterBuilder.build_reader(io_name)
    writer = IOAdapterBuilder.build_writer(io_name)
    provider = ProviderAdapterBuilder.build(
        provider_name=provider_name,
        model_config=model_config,
        retry_config=retry_config,
    )

    # Create domain service
    service = InferenceService(
        example_reader=reader,
        judgement_writer=writer,
        prompt_builder=prompt_builder,
        llm_provider=provider,
        response_parser=response_parser,
    )

    # Run inference pipeline
    try:
        summary = service.run_inference(
            run_name=input_run_name,
            model_config=model_config,
            run_info=run_info,
            run_dir=run_dir,
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
