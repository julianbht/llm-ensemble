"""CLI driving adapter for inference pipeline.

Driving Adapter Layer - CLI Infrastructure

This adapter wraps the inference application use case and provides CLI-specific infrastructure:
- File-based run directory management
- Terminal logging with structured events
- File-based result persistence (summary.json)
- Progress reporting to stdout

Follows hexagonal architecture pattern where the driver:
1. Receives the application's driving port (InferenceUseCase) in constructor
2. Provides infrastructure specific to the CLI execution context
3. Executes the application via its public interface
4. Handles results in a CLI-appropriate manner

Comparable to ForParkingCarsTestDriver in BlueZone example.
Tested via CLI integration tests.
"""
from __future__ import annotations
from typing import Tuple
from logging import Logger
from pathlib import Path

from llm_ensemble.infer.application.inference_use_case import InferenceUseCase
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.startup.adapter_config import ExecutionParams
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary

from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager


class CLIDriver:
    """CLI driving adapter for inference pipeline.

    Wraps the inference application and provides CLI-specific infrastructure concerns.
    The application exposes its driving port (InferenceUseCase.execute), which this
    adapter uses to execute the business logic while handling CLI-specific concerns
    like file-based logging, run directories, and terminal output.

    Attributes:
        application: The inference use case (application's driving port)
        run_config: Domain configuration for the inference run
        execution_params: CLI execution parameters (run name, official flag, etc.)
        logging_config_name: Name of logging config to load
    """

    def __init__(
        self,
        application: InferenceUseCase,
        run_config: InferRunConfig,
        execution_params: ExecutionParams,
        logging_config_name: str,
    ):
        """Initialize CLI driver with application and CLI-specific configuration.

        Args:
            application: The inference use case (application's driving port)
            run_config: Domain configuration bundle
            execution_params: CLI execution parameters
            logging_config_name: Name of logging config file
        """
        self.application = application
        self.run_config = run_config
        self.execution_params = execution_params
        self.logging_config_name = logging_config_name

    def run(self) -> None:
        """Execute inference pipeline with CLI-specific infrastructure.

        Flow:
        1. Setup CLI infrastructure (run dir, file-based logging)
        2. Execute application use case (pure business logic)
        3. Finalize CLI outputs (write summary, log completion)
        """
        # CLI-specific: setup file-based infrastructure
        run_info, logger = self._setup_infrastructure()

        # Execute application (pure business logic via driving port)
        summary = self.application.execute(
            run_info=run_info,
            run_config=self.run_config,
        )

        # CLI-specific: finalize results
        self._finalize_run(summary, run_info, logger)

    def _setup_infrastructure(self) -> Tuple[InferRunInfo, Logger]:
        """Setup CLI-specific infrastructure: run directory and file-based logging.

        Returns:
            Tuple of (run_info, logger)
        """
        logging_config = LoggingConfig.load(self.logging_config_name)

        # Generate run name and create run directory
        name_hints = [
            self.run_config.model_cfg.name_hint,
            self.run_config.prompt_template.name,
            self.run_config.provider.name,
        ]
        run_info = InferRunInfo.create(
            name_hints=name_hints,
            run_name=self.execution_params.run_name,
            official=self.execution_params.official,
            notes=self.execution_params.notes,
        )

        # Create run directory (CLI-specific: file-based output)
        run_dir = run_info.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create tag symlink if requested (CLI-specific)
        if self.execution_params.tag:
            TagManager.create_tag(run_dir, self.execution_params.tag)

        # Setup file-based logging (CLI-specific)
        log_file = run_dir / "run.log" if logging_config.save_logs else None
        logger = configure_logger(
            cli_name="infer",
            run_name=self.execution_params.run_name,
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
            model=self.run_config.model_cfg.name_hint,
            provider=self.run_config.provider.name,
            io_format=self.run_config.io_name,
            prompt_template=self.run_config.prompt_template.name,
            input_run_name=self.run_config.ingest_run_context.input_run_name,
            start_idx=self.run_config.ingest_run_context.start_idx,
            end_idx=self.run_config.ingest_run_context.end_idx,
        )

        return run_info, logger

    def _finalize_run(
        self,
        summary: InferRunSummary,
        run_info: InferRunInfo,
        logger: Logger,
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
