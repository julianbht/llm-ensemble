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

import os
from typing import Optional
import structlog

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.runtime.run_info import RunType
from pathlib import Path
from llm_ensemble.libs.runtime.run_name import generate_run_name

from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory

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
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder, write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.schemas.logging_config import LoggingConfig

# Load runtime configuration (.env file)
from llm_ensemble.libs.runtime.env import load_runtime_config
load_runtime_config()


class InferenceApplication(ForRunningInference):
    """Application use case for coordinating LLM inference pipeline.

    Implements the driving port ForRunningInference - this IS the application's API.
    Driving adapters (CLI, Web API, etc.) call the execute() method.

    This is the backend application that handles:
    - Infrastructure setup (run directories, logging configuration)
    - Inference execution (read → prompt → infer → parse → write loop)
    - Result persistence and finalization

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
    ):
        """Initialize inference use case with port dependencies.

        Args:
            input_port: Port for reading input data
            output_port: Port for writing output data
            prompt_builder: Port for building prompts from samples
            llm_provider: Port for LLM inference
            response_parser: Port for parsing LLM responses
        """
        self.input_port = input_port
        self.output_port = output_port
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.response_parser = response_parser

    def run_inference(
        self,
        input_run_name: str,
        start_idx: Optional[int],
        end_idx: Optional[int],
        run_name: Optional[str],
        official: bool,
        notes: Optional[str],
        tag: Optional[str],
    ) -> InferRunSummary:
        """Execute the complete inference backend with infrastructure setup and finalization.

        This is the main application entry point called by driving adapters (CLI, Web API).

        Backend workflow:
        - Setup infrastructure (run directories, logging, metadata)
        - Read dataset samples via InputPort
        - For each sample: build prompt → infer → parse → write (streaming loop)
        - Write summary and finalize outputs
        - Return summary statistics

        All logging configured here appears in the driving adapter's output
        (terminal for CLI, CloudWatch for web API, etc.).

        Args:
            input_run_name: Ingest run identifier to read samples from
            start_idx: Start index into NormalizedDataset (None = from beginning)
            end_idx: End index into NormalizedDataset (None = until end)
            run_name: Custom run name (auto-generates if not provided)
            official: Mark as official run
            notes: Notes about this run (experiment purpose, hypothesis, etc.)
            tag: Tag name for easy reference by downstream CLIs

        Returns:
            InferRunSummary with statistics, timing, and warnings

        Raises:
            Exception: If any step in the pipeline fails
        """
        # Setup
        run_name = self._generate_run_name(run_name)
        run_dir = self._create_run_directory(run_name, official, tag)
        logger = self._setup_logging(run_name, run_dir)

        # Log startup
        logger.info(
            InferLogEvent.INFER_STARTED,
            model=self.llm_provider.model_config.name_hint,
            provider=self.llm_provider.provider_name,
            io_format=self.output_port.io_name,
            input_run_name=input_run_name,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # Execute inference pipeline
        summary_builder = RunSummaryBuilder()
        summary_builder.set_start_time()

        # Resolve input run name (handles tags)
        resolved_input_run_name = TagManager.resolve_input(input_run_name, "ingest")

        # Read full NormalizedDataset
        normalized_dataset = self.input_port.read(resolved_input_run_name)

        # Compute actual start_idx and end_idx
        actual_start_idx = start_idx if start_idx is not None else 0
        actual_end_idx = end_idx if end_idx is not None else len(normalized_dataset.samples)

        # Slice samples based on computed indices
        samples_to_process = normalized_dataset.samples[actual_start_idx:actual_end_idx]

        # Collect judgements for summary statistics
        llm_judgements: list[LLMJudgement] = []

        # Build run_config and run_info for manifest/persistence
        run_config = self._build_run_config(resolved_input_run_name, actual_start_idx, actual_end_idx)
        run_info = self._build_run_info(run_name, official, notes)

        # Open writer for streaming
        with self.output_port.open(run_dir, run_info, run_config, normalized_dataset) as writer:

            # Process each dataset sample in slice (streaming loop)
            for dataset_sample in samples_to_process:

                # Build prompt text
                prompt_text = self.prompt_builder.build_prompt(dataset_sample)

                # Run inference (model_config was passed at provider initialization)
                logger.info(InferLogEvent.SENDING_REQUEST)
                raw_response_text, llm_invocation_metrics = self.llm_provider.infer(prompt_text)

                # Parse response (returns tuple of score and warnings)
                llm_score, parser_warnings = self.response_parser.parse(raw_response_text)

                # Create judgement with flat structure (no configs)
                judgement = LLMJudgement(
                    dataset_sample=dataset_sample,
                    prompt_text=prompt_text,
                    response_text=raw_response_text,
                    llm_invocation_metrics=llm_invocation_metrics,
                    llm_score=llm_score,
                    parser_warnings=parser_warnings,
                )

                # Log each completed judgement
                extracted_score = judgement.llm_score.label.value if judgement.llm_score and judgement.llm_score.label else "null"
                gold_score = judgement.dataset_sample.judging_sample.gold_score.value
                latency_s = judgement.llm_invocation_metrics.latency_ms / 1000
                logger.info(
                    InferLogEvent.RESPONSE_PARSED,
                    extracted_score=extracted_score,
                    gold_score=gold_score,
                    latency_s=f"{latency_s:.1f}",
                )

                # Log metrics for observability dashboard (cost, agreement, latency)
                cost_usd = judgement.llm_invocation_metrics.cost_estimate_usd or 0.0
                agreement = 1 if extracted_score == gold_score else 0
                logger.info(
                    InferLogEvent.JUDGEMENT_METRICS,
                    cost_estimate_usd=cost_usd,
                    agreement=agreement,
                    latency_s=latency_s,
                )

                # Write judgement immediately to disk (fault tolerance!)
                writer.write_one(judgement)

                # Collect for summary statistics
                llm_judgements.append(judgement)

        # Retrieve aggregate write summary after context manager closes (writer logged directly)
        write_summary = self.output_port.get_summary()

        # Calculate aggregate statistics from judgements (for summary)
        count = len(llm_judgements)
        error_count = sum(1 for j in llm_judgements if j.llm_score is None or j.llm_score.label is None)
        total_latency_ms = sum(j.llm_invocation_metrics.latency_ms for j in llm_judgements)
        avg_latency = total_latency_ms / count if count > 0 else 0.0

        # Build warnings summary from all judgements
        warnings_summary: dict[str, int] = {}
        for judgement in llm_judgements:
            # Aggregate parser warnings from all judgements
            for warning in judgement.parser_warnings:
                warning_type = warning.__class__.__name__
                warnings_summary[warning_type] = warnings_summary.get(warning_type, 0) + 1

        # Add write summary to builder for inclusion in final summary
        summary_builder.add("write_summary", write_summary)

        # Add statistics to summary builder
        summary_builder.add("judgement_count", count)
        summary_builder.add("error_count", error_count)
        summary_builder.add("total_latency_ms", total_latency_ms)
        summary_builder.add("avg_latency_ms", avg_latency)

        # Add warnings summary to builder
        if warnings_summary:
            summary_builder.add("warnings_summary", warnings_summary)

        # Finalize summary (sets end_time and creates immutable Pydantic object)
        summary: InferRunSummary = summary_builder.finalize(InferRunSummary)

        # Finalize outputs (write summary, log completion)
        self._finalize_run(summary, run_dir, logger)

        # Return finalized summary
        return summary

    def _build_run_config(
        self,
        input_run_name: str,
        start_idx: int,
        end_idx: int,
    ) -> InferRunConfig:
        """Build run config for manifest by querying adapter config.

        Args:
            input_run_name: Resolved ingest run name
            start_idx: Actual start index used
            end_idx: Actual end index used

        Returns:
            InferRunConfig for manifest persistence
        """
        # Query config from adapters
        model_config = self.llm_provider.model_config
        retry_config = self.llm_provider.retry_config
        provider = self.llm_provider.get_provider()
        prompt_builder = self.prompt_builder.get_builder()
        response_parser = self.response_parser.get_parser()
        io_name = self.output_port.io_name

        # Build config entity
        return InferRunConfig(
            model_cfg=model_config,
            retry_config=retry_config,
            prompt_template=PromptTemplateFactory.create(prompt_builder.name),
            provider=provider,
            io_name=io_name,
            ingest_run_context=IngestRunContext(
                input_run_name=input_run_name,
                start_idx=start_idx,
                end_idx=end_idx,
            ),
        )

    def _generate_run_name(self, run_name: Optional[str]) -> str:
        """Generate run name from adapter configs if not provided.

        Args:
            run_name: Custom run name or None to auto-generate

        Returns:
            Run name (either custom or auto-generated)
        """
        if run_name is not None:
            return run_name

        # Generate from adapter configs
        name_hints: list[str] = [
            self.llm_provider.model_config.name_hint,
            self.prompt_builder.get_builder().name,
            self.llm_provider.get_provider().name,
        ]
        return generate_run_name(name_hints)

    def _create_run_directory(
        self,
        run_name: str,
        official: bool,
        tag: Optional[str],
    ) -> Path:
        """Create run directory and optional tag symlink.

        Args:
            run_name: Run name for directory
            official: Whether this is an official run
            tag: Optional tag name for symlink

        Returns:
            Path to run directory
        """
        run_type = RunType.OFFICIAL if official else RunType.TEST
        run_dir = PathManager.get_run_dir("infer", run_name, run_type.value)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create tag symlink if requested
        if tag:
            TagManager.create_tag(run_dir, tag)

        return run_dir

    def _setup_logging(
        self,
        run_name: str,
        run_dir: "Path",
    ) -> structlog.stdlib.BoundLogger:
        """Configure logging for this inference run.

        Args:
            run_name: Run name for logger context
            run_dir: Run directory for log file

        Returns:
            Configured structlog logger instance
        """
        # Load logging config from environment variable or use default
        logging_config_name = os.getenv("LOGGING_CONFIG", "observability")
        logging_config = LoggingConfig.load(logging_config_name)

        # Setup logging (configured output appears in driving adapter: terminal for CLI, CloudWatch for web, etc.)
        log_file = run_dir / "run.log" if logging_config.save_logs else None
        logger = configure_logger(
            cli_name="infer",
            run_name=run_name,
            run_type="test",  # Default, will be in run_info for manifest
            pretty_print=logging_config.pretty_print,
            save_logs=logging_config.save_logs,
            log_file_path=log_file,
            console_level=logging_config.console_level,
            file_level=logging_config.file_level,
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

    def _finalize_run(
        self,
        summary: InferRunSummary,
        run_dir: "Path",
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        """Finalize backend outputs: write summary to disk and log completion.

        Args:
            summary: Inference run summary from pipeline execution
            run_dir: Run directory path
            logger: Configured logger instance
        """
        # Log completion
        logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=summary.judgement_count)

        # Write summary to disk
        write_standalone_summary(summary, run_dir)
        logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(run_dir / "summary.json"))

        # Log final statistics
        logger.info(
            InferLogEvent.INFER_COMPLETE,
            total_judgements=summary.judgement_count,
            parsing_failures=summary.error_count,
            avg_latency_ms=f"{summary.avg_latency_ms:.1f}",
        )

        # Log warnings if any
        if summary.warnings_summary and sum(summary.warnings_summary.values()) > 0:
            total_warnings = sum(summary.warnings_summary.values())
            logger.info(
                InferLogEvent.WARNINGS_COLLECTED,
                total_warnings=total_warnings,
                **summary.warnings_summary
            )

        # Log file location
        log_file = run_dir / "run.log"
        if log_file.exists():
            logger.info(InferLogEvent.LOGS_SAVED, path=str(log_file))
