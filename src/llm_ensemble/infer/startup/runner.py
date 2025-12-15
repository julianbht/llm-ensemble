"""Inference pipeline runner.

Startup Layer - Infrastructure Orchestration (BlueZoneRunner equivalent)

Main entry point for inference execution. Responsible for:
1. Receiving lightweight request configs from CLI (AdapterConfig, ExecutionParams)
2. Loading full configurations (heavyweight config objects)
3. Setting up infrastructure (run directories, logging)
4. Creating dependency configurator
5. EXPLICITLY looking up adapters (visible dependency graph!)
6. Building InferRunConfig domain entity (separate from adapter instantiation)
7. Building and executing application with explicit dependencies
8. Post-processing (summaries, manifests)

This is the composition root - NOT unit tested.
Tested via CLI integration tests.

Key pattern (from BlueZone):
- CLI builds lightweight request objects (adapter selection, execution params)
- Runner loads configs and instantiates adapters
- Adapter lookup is EXPLICIT in runner (not hidden in configurator)
- build_application() takes EXPLICIT port arguments (dependencies visible!)
- InferRunConfig is a domain entity (for use case), not for adapter instantiation
"""
from __future__ import annotations

from llm_ensemble.infer.startup.adapter_config import AdapterConfig, ExecutionParams
from llm_ensemble.infer.startup.dependency_configurator import DependencyConfigurator
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory

from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager


def run_inference(
    adapter_config: AdapterConfig,
    execution_params: ExecutionParams,
) -> None:
    """Run inference pipeline with full infrastructure setup.

    Main orchestration function (BlueZoneRunner.main() equivalent).

    Pattern:
    1. Receive lightweight request configs from CLI
    2. Load heavyweight config objects
    3. Create DependencyConfigurator with loaded configs
    4. EXPLICITLY lookup adapters (visible in runner!)
    5. Build InferRunConfig domain entity (separate concern)
    6. Build application with explicit port arguments
    7. Execute use case

    Args:
        adapter_config: Adapter selection config (which adapters to use)
        execution_params: Execution parameters (how to execute, run metadata)

    Raises:
        FileNotFoundError: If config or input run doesn't exist
        ValueError: If adapter is not recognized or config is invalid
    """
    # Resolve tag if needed
    input_run_name = TagManager.resolve_input(execution_params.input_run_name, "ingest")

    # ========================================================================
    # STEP 1: Load Heavyweight Configurations (Loaded Config Objects)
    # ========================================================================

    # Load individual config objects from YAML files
    model_config = ModelConfig.load(adapter_config.model_config_name)
    retry_config = RetryConfig.load(adapter_config.retry_config_name)
    prompt_template = PromptTemplateFactory.create(adapter_config.prompt_template_name)
    logging_config = LoggingConfig.load(adapter_config.logging_config_name)

    # ========================================================================
    # STEP 2: Build Domain Entity (InferRunConfig)
    # ========================================================================
    # This is separate from adapter instantiation!
    # It's a domain entity for use case provenance.

    run_config = InferRunConfig(
        model_cfg=model_config,
        provider=Provider(name=adapter_config.provider_name),
        prompt_template=prompt_template,
        retry_config=retry_config,
        ingest_run_context=IngestRunContext(
            input_run_name=input_run_name,
            start_idx=execution_params.start_idx,
            end_idx=execution_params.end_idx,
        ),
    )

    # ========================================================================
    # STEP 3: Setup Infrastructure
    # ========================================================================

    # Create run info (metadata)
    run_info = InferRunInfo.create(
        name_hints=run_config.get_name_hints(),
        run_name=execution_params.run_name,
        official=execution_params.official,
        notes=execution_params.notes,
    )

    # Setup run directory
    run_dir = run_info.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create tag if provided
    if execution_params.tag:
        TagManager.create_tag(run_dir, execution_params.tag)

    # Setup logger
    log_file_path = run_dir / "run.log" if logging_config.save_logs else None
    logger = configure_logger(
        cli_name="infer",
        run_name=execution_params.run_name,
        run_type=run_info.run_type,
        pretty_print=logging_config.pretty_print,
        save_logs=logging_config.save_logs,
        log_file_path=log_file_path,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
    )

    logger.info(
        InferLogEvent.INFER_STARTED,
        model=adapter_config.model_config_name,
        provider=adapter_config.provider_name,
        io_format=adapter_config.io_name,
        prompt_template=adapter_config.prompt_template_name,
        input_run_name=input_run_name,
        start_idx=execution_params.start_idx,
        end_idx=execution_params.end_idx,
    )

    # ========================================================================
    # STEP 4: Create Dependency Configurator
    # ========================================================================

    dependency_configurator = DependencyConfigurator(
        model_config=model_config,
        provider_name=adapter_config.provider_name,
        prompt_template=prompt_template,
        retry_config=retry_config,
    )

    # ========================================================================
    # STEP 5: EXPLICITLY Lookup Adapters (BlueZone Pattern!)
    # ========================================================================
    # This makes the dependency graph VISIBLE in the runner!
    # You can see exactly what adapters are being used.

    try:
        # Lookup driven ports (explicit!)
        input_port = dependency_configurator.lookup_input_port(adapter_config.io_name)
        output_port = dependency_configurator.lookup_output_port(adapter_config.io_name)
        prompt_builder = dependency_configurator.lookup_prompt_builder()
        response_parser = dependency_configurator.lookup_response_parser()
        llm_provider = dependency_configurator.lookup_llm_provider()

        # ========================================================================
        # STEP 6: Build Application with EXPLICIT Dependencies
        # ========================================================================
        # build_application() takes explicit arguments - dependencies are VISIBLE!

        use_case = dependency_configurator.build_application(
            input_port=input_port,
            output_port=output_port,
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            llm_provider=llm_provider,
        )

        # ========================================================================
        # STEP 7: Execute Use Case
        # ========================================================================

        summary = use_case.execute(
            run_info=run_info,
            run_config=run_config,
        )

        # ========================================================================
        # STEP 8: Post-Processing
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
