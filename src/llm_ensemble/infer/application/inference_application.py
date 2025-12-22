"""Application for LLM inference pipeline.

Application Layer - Hexagonal Architecture

This module contains the complete inference backend orchestration.
It implements the driving port (ForRunningInference) and handles:
- Infrastructure setup (run directories, logging)
- Inference pipeline execution via driven ports
- Result persistence and finalization

This is the backend that both CLI and Web adapters use.
Depends only on port abstractions for testability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import structlog

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.llm_judgement_builder import LLMJudgementBuilder
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.metrics import (
    calculate_agreement,
    calculate_aggregate_statistics,
    calculate_latency_seconds,
    get_extracted_score,
)
from llm_ensemble.infer.domain.entities.infer_run_summary import InferRunSummary
from llm_ensemble.infer.application.write_summary import WriteSummary
from llm_ensemble.infer.domain.infer_run_config_factory import InferRunConfigFactory
from llm_ensemble.infer.domain.dataset_utils import slice_dataset, resolve_slice_indices

# Driving port (application implements this)
from llm_ensemble.infer.application.ports.driving.for_running_inference import ForRunningInference

# Driven ports (application depends on these)
from llm_ensemble.infer.application.ports.driven.llm_provider_port import LLMProviderPort
from llm_ensemble.infer.application.ports.driven.input_port import InputPort
from llm_ensemble.infer.application.ports.driven.output_port import OutputPort
from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort
from llm_ensemble.infer.application.ports.driven.prompt_builder_port import PromptBuilderPort

from llm_ensemble.libs.logging import configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.runtime.run_manager import write_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager

# Load runtime env configuration
from llm_ensemble.libs.runtime.env import load_runtime_config
load_runtime_config()


class InferenceApplication(ForRunningInference):
    """Application use case for coordinating LLM inference pipeline.

    Implements the driving port ForRunningInference - this IS the application's API.
    Driving adapters (CLI, Web API, etc.) call the execute() method.

    This is the backend application that handles:
    - Logging configuration
    - Inference execution (read → prompt → infer → parse → write loop)
    - Result persistence and finalization

    Infrastructure setup (run directories, run naming) is handled by the
    composition root before application instantiation.

    Driving adapters are thin wrappers that:
    - Parse input (CLI args, HTTP requests)
    - Call this application
    - Present results (terminal output, HTTP responses)

    Depends only on driven port abstractions, enabling unit testing with mocked ports.
    """

    def __init__(
        self,
        input_port: InputPort,
        output_port: OutputPort,
        prompt_builder: PromptBuilderPort,
        llm_provider: LLMProviderPort,
        response_parser: ResponseParserPort,
        run_dir: Path,
        run_name: str,
    ):
        """Initialize inference use case with port dependencies.

        Args:
            input_port: Port for reading input data
            output_port: Port for writing output data
            prompt_builder: Port for building prompts from samples
            llm_provider: Port for LLM inference
            response_parser: Port for parsing LLM responses
            run_dir: Run directory (created by composition root)
            run_name: Run name (resolved by composition root)
        """
        self.input_port = input_port
        self.output_port = output_port
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.response_parser = response_parser
        self.run_dir = run_dir
        self.run_name = run_name

    def run_inference(
        self,
        input_run_name: str,
        start_idx: Optional[int],
        end_idx: Optional[int],
        official: bool,
        notes: Optional[str],
    ) -> InferRunSummary:
        """Execute the complete inference backend with infrastructure setup and finalization.

        This is the main application entry point called by driving adapters (CLI, Web API).

        Backend workflow:
        - Setup logging
        - Read dataset samples via InputPort
        - For each sample: build prompt → infer → parse → write (streaming loop)
        - Write summary and finalize outputs
        - Return summary statistics

        Run directory and run name are provided at construction time by the
        composition root.

        All logging configured here appears in the driving adapter's output
        (terminal for CLI, CloudWatch for web API, etc.).

        Args:
            input_run_name: Ingest run identifier to read samples from
            start_idx: Start index into NormalizedDataset (None = from beginning)
            end_idx: End index into NormalizedDataset (None = until end)
            official: Mark as official run
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            InferRunSummary with statistics, timing, and warnings

        Raises:
            Exception: If any step in the pipeline fails
        """
        # Track start time for summary
        start_time = datetime.now()

        # Setup logging (run_dir and run_name provided by composition root)
        logger = self._setup_logging(self.run_name, self.run_dir)

        logger.info(
            InferLogEvent.INFER_STARTED,
            run_name=self.run_name
        )

        # Read full NormalizedDataset
        resolved_input_run_name = TagManager.resolve_input(input_run_name, "ingest")
        normalized_dataset = self.input_port.read(resolved_input_run_name)

        # Slice to get samples that we want to judge
        actual_start_idx, actual_end_idx = resolve_slice_indices(normalized_dataset, start_idx, end_idx)
        samples_to_process = normalized_dataset.samples[actual_start_idx:actual_end_idx]

        # Collect judgements for summary statistics
        llm_judgements: list[LLMJudgement] = []

        # Build run_config and run_info for manifest/persistence
        run_config = self._build_run_config(resolved_input_run_name, actual_start_idx, actual_end_idx)
        run_info = self._build_run_info(self.run_name, official, notes)

        # Open writer for streaming (run_dir already provided at construction)
        with self.output_port.open(run_info, run_config) as writer:

            # Process each dataset sample in slice (streaming loop)
            for dataset_sample in samples_to_process:

                # Initialize builder for this sample
                builder = LLMJudgementBuilder(dataset_sample)

                # Step 1: Build prompt
                prompt_text = self.prompt_builder.build_prompt(dataset_sample)
                builder.with_prompt(prompt_text)

                # Step 2: Run inference (model_config was passed at provider initialization)
                raw_response_text, llm_invocation_metrics = self.llm_provider.infer(prompt_text)
                builder.with_llm_response(raw_response_text, llm_invocation_metrics)

                # Step 3: Parse response (returns tuple of score and warnings)
                llm_score, parser_warnings = self.response_parser.parse(raw_response_text)
                builder.with_parsed_score(llm_score, parser_warnings)

                # Build complete judgement
                judgement = builder.build()

                logger.info(
                    InferLogEvent.JUDGEMENT_PROCESSED,
                    extracted_score=get_extracted_score(judgement),
                    gold_score=judgement.dataset_sample.judging_sample.gold_score.value,
                    agreement=calculate_agreement(judgement),
                    latency_s=calculate_latency_seconds(judgement),
                )

                # Write judgement immediately to disk (fault tolerance!)
                writer.write_one(judgement)

                # Collect for summary statistics
                llm_judgements.append(judgement)

        logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=llm_judgements.count)

        # Retrieve aggregate write summary after context manager closes (writer logged directly)
        write_summary = self.output_port.get_summary()

        # Build and finalize summary
        summary = self._build_summary(start_time, llm_judgements, write_summary)
        self._finalize_run(summary, self.run_dir, logger)

        # Return finalized summary
        return summary

    def _build_run_config(
        self,
        input_run_name: str,
        start_idx: int,
        end_idx: int,
    ) -> InferRunConfig:
        """Build run config for manifest by extracting entities from adapters.

        Application layer responsibility: extract domain entities from ports,
        then delegate to domain factory for assembly.

        Args:
            input_run_name: Resolved ingest run name
            start_idx: Actual start index used
            end_idx: Actual end index used

        Returns:
            InferRunConfig for manifest persistence
        """
        # Extract domain entities from adapters
        provider = self.llm_provider.get_provider()
        model_config = self.llm_provider.get_model_config()
        retry_config = self.llm_provider.get_retry_config()
        prompt_builder_entity = self.prompt_builder.get_builder()
        response_parser_entity = self.response_parser.get_parser()
        template_text = self.prompt_builder.get_template_text()
        io_name = self.output_port.io_name

        # Delegate to domain factory for assembly
        return InferRunConfigFactory.create(
            provider=provider,
            model_config=model_config,
            retry_config=retry_config,
            prompt_builder=prompt_builder_entity,
            response_parser=response_parser_entity,
            template_text=template_text,
            io_name=io_name,
            input_run_name=input_run_name,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def _setup_logging(
        self,
        run_name: str,
        run_dir: "Path",
    ) -> structlog.stdlib.BoundLogger:
        """Configure logging for this inference run.

        Logging configuration is read from environment variables:
            LOG_PRETTY_PRINT: Use human-readable console output (true/false, default: false)
            LOG_SAVE_LOGS: Save logs to file (true/false, default: true)
            LOG_CONSOLE_LEVEL: Console log level (DEBUG/INFO/WARNING/ERROR, default: INFO)
            LOG_FILE_LEVEL: File log level (DEBUG/INFO/WARNING/ERROR, default: DEBUG)

        Args:
            run_name: Run name for logger context
            run_dir: Run directory for log file

        Returns:
            Configured structlog logger instance
        """
        # Setup logging from environment variables
        logger = configure_logger(
            cli_name="infer",
            run_name=run_name,
            run_type="test",
            log_file_path=run_dir / "run.log",
        )

        return logger

    def _build_run_info(
        self,
        run_name: str,
        official: bool,
        notes: Optional[str],
    ) -> InferRunInfo:
        """Build InferRunInfo entity for manifest persistence.

        Args:
            run_name: Run name
            official: Whether this is an official run
            notes: Optional notes about this run

        Returns:
            InferRunInfo entity
        """
        return InferRunInfo(
            run_name=run_name,
            run_type=RunType.OFFICIAL if official else RunType.TEST,
            notes=notes,
        )

    def _build_summary(
        self,
        start_time: datetime,
        llm_judgements: list[LLMJudgement],
        write_summary: WriteSummary,
    ) -> InferRunSummary:
        """Build run summary from execution results.

        Args:
            start_time: Time when inference pipeline started
            llm_judgements: List of all judgements produced
            write_summary: Write operation summary from output port

        Returns:
            Complete InferRunSummary with all statistics
        """
        # Calculate domain statistics
        (
            judgement_count,
            error_count,
            total_latency_ms,
            avg_latency_ms,
            warnings_summary,
        ) = calculate_aggregate_statistics(llm_judgements)
        
        # Construct Pydantic summary (application responsibility)
        return InferRunSummary(
            start_time=start_time,
            end_time=datetime.now(),
            write_summary=write_summary,
            judgement_count=judgement_count,
            error_count=error_count,
            total_latency_ms=total_latency_ms,
            avg_latency_ms=avg_latency_ms,
            warnings_summary=warnings_summary,
        )

    def _finalize_run(
        self,
        summary: InferRunSummary,
        run_dir: Path,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        """Finalize backend outputs: write summary to disk and log completion.

        Args:
            summary: Inference run summary from pipeline execution
            run_dir: Run directory path
            logger: Configured logger instance
        """
        # Write summary to disk
        summary_path = write_summary(summary, run_dir)
        logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(summary_path))

        # Log file location
        log_file = run_dir / "run.log"
        if log_file.exists():
            logger.info(InferLogEvent.LOGS_SAVED, path=str(log_file))
