"""Inference pipeline runner.

Startup Layer - Infrastructure Orchestration (BlueZoneRunner equivalent)

Main entry point for inference execution. Responsible for:
1. Loading configurations
2. Setting up infrastructure (run directories, logging)
3. Creating dependency configurator
4. Building and executing application
5. Post-processing (summaries, manifests)

This is the composition root - NOT unit tested.
Tested via CLI integration tests.
"""
from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.startup.dependency_configurator import DependencyConfigurator
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo

from llm_ensemble.infer.application.config_builder import build_infer_config

from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager


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
    """Run inference pipeline with full infrastructure setup.

    Main orchestration function (BlueZoneRunner.main() equivalent).

    Args:
        model_config_name: Name of the model config file
        provider_name: Provider name for registry lookup
        prompt_template_name: Prompt template name (bundles builder and parser)
        retry_config_name: Name of the retry config file
        io_name: I/O format name
        input_run_name: Ingest run identifier
        logging_config_name: Name of the logging config file
        run_name: Custom run ID
        start_idx: Start index into NormalizedDataset
        end_idx: End index into NormalizedDataset
        official: Mark as official run
        notes: Notes about this run
        tag: Tag name for easy reference

    Raises:
        FileNotFoundError: If config or input run doesn't exist
        ValueError: If adapter is not recognized or config is invalid
    """
    # Resolve tag if needed
    input_run_name = TagManager.resolve_input(input_run_name, "ingest")

    # ========================================================================
    # STEP 1: Load Configurations
    # ========================================================================

    run_config = build_infer_config(
        model_config_name=model_config_name,
        provider_name=provider_name,
        prompt_template_name=prompt_template_name,
        retry_config_name=retry_config_name,
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    logging_config = LoggingConfig.load(logging_config_name)

    # ========================================================================
    # STEP 2: Setup Infrastructure
    # ========================================================================

    # Create run info (metadata)
    run_info = InferRunInfo.create(
        name_hints=run_config.get_name_hints(),
        run_name=run_name,
        official=official,
        notes=notes,
    )

    # Setup run directory
    run_dir = run_info.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create tag if provided
    if tag:
        TagManager.create_tag(run_dir, tag)

    # Setup logger
    log_file_path = run_dir / "run.log" if logging_config.save_logs else None
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

    # ========================================================================
    # STEP 3: Create Dependency Configurator
    # ========================================================================

    dependency_configurator = DependencyConfigurator(
        run_config=run_config,
        io_name=io_name,
    )

    # ========================================================================
    # STEP 4: Build Application and Execute
    # ========================================================================

    try:
        # Build application (hexagon) with injected dependencies
        use_case = dependency_configurator.build_application()

        # Execute use case
        summary = use_case.execute(
            run_info=run_info,
            run_config=run_config,
        )

        # ========================================================================
        # STEP 5: Post-Processing
        # ========================================================================

        logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=summary.judgement_count)

        # Write standalone summary.json for convenience
        write_standalone_summary(summary, run_dir)
        logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(run_dir / "summary.json"))

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

    except Exception as e:
        logger.error(InferLogEvent.INFER_FAILED, error=str(e))
        raise
