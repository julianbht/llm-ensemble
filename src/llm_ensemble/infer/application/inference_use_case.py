"""Application use case for LLM inference pipeline.

Application Layer - Use Case

This module contains the complete inference backend orchestration.
It implements the driving port (ForRunningInference) and handles:
- Infrastructure setup (run directories, logging)
- Inference pipeline execution via driven ports
- Result persistence and finalization

This is the backend that both CLI and Web adapters use.
Depends only on port abstractions for testability.
"""

from __future__ import annotations

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary
from llm_ensemble.infer.startup.adapter_config import ExecutionParams

# Driving port (application implements this)
from llm_ensemble.infer.application.ports.driving.for_running_inference import ForRunningInference

# Driven ports (application depends on these)
from llm_ensemble.infer.application.ports.driven.llm_provider_port import LLMProviderPort
from llm_ensemble.infer.application.ports.driven.input_port import InputPort
from llm_ensemble.infer.application.ports.driven.output_port import OutputPort
from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort
from llm_ensemble.infer.application.ports.driven.prompt_builder_port import PromptBuilderPort

from llm_ensemble.libs.logging import get_logger, configure_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder, write_standalone_summary
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.schemas.logging_config import LoggingConfig


class InferenceUseCase(ForRunningInference):
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

    def execute(
        self,
        run_config: InferRunConfig,
        execution_params: ExecutionParams,
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
            run_config: Configuration bundle (model, provider, adapters, execution context)
            execution_params: Execution parameters (run name, official flag, notes, tag)

        Returns:
            InferRunSummary with statistics, timing, and warnings

        Raises:
            Exception: If any step in the pipeline fails
        """
        # Setup infrastructure (run directories, logging, git metadata)
        run_info, logger = self._setup_infrastructure(run_config, execution_params)

        # Execute inference pipeline
        summary_builder = RunSummaryBuilder()
        summary_builder.set_start_time()

        # Derive run_dir from run_info
        run_dir = run_info.run_dir

        # Read full NormalizedDataset (using input_run_name from execution context)
        normalized_dataset = self.input_port.read(run_config.ingest_run_context.input_run_name)

        # Compute actual start_idx and end_idx from execution context
        start_idx = run_config.ingest_run_context.start_idx if run_config.ingest_run_context.start_idx is not None else 0
        end_idx = run_config.ingest_run_context.end_idx if run_config.ingest_run_context.end_idx is not None else len(normalized_dataset.samples)

        # Slice samples based on computed indices
        samples_to_process = normalized_dataset.samples[start_idx:end_idx]

        # Collect judgements for summary statistics
        llm_judgements: list[LLMJudgement] = []

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
        self._finalize_run(summary, run_info, logger)

        # Return finalized summary
        return summary

    def _setup_infrastructure(
        self,
        run_config: InferRunConfig,
        execution_params: ExecutionParams,
    ) -> tuple[InferRunInfo, object]:
        """Setup backend infrastructure: run directories, logging, git metadata.

        Loads logging config from environment or defaults, creates run directories,
        configures logging for this run, and logs startup information.

        Args:
            run_config: Domain configuration bundle
            execution_params: Execution parameters

        Returns:
            Tuple of (run_info, logger)
        """
        # Load logging config from environment or use default
        logging_config = LoggingConfig.load("observability")

        # Generate run name and create run metadata
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

        # Create run directory
        run_dir = run_info.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create tag symlink if requested
        if execution_params.tag:
            TagManager.create_tag(run_dir, execution_params.tag)

        # Setup logging (configured output appears in driving adapter: terminal for CLI, CloudWatch for web, etc.)
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

        # Log startup information
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
        self,
        summary: InferRunSummary,
        run_info: InferRunInfo,
        logger: object,
    ) -> None:
        """Finalize backend outputs: write summary to disk and log completion.

        Args:
            summary: Inference run summary from pipeline execution
            run_info: Run metadata
            logger: Configured logger instance
        """
        # Log completion
        logger.info(InferLogEvent.ALL_SAMPLES_PROCESSED, count=summary.judgement_count)

        # Write summary to disk
        write_standalone_summary(summary, run_info.run_dir)
        logger.info(InferLogEvent.INFER_SUMMARY_WRITTEN, path=str(run_info.run_dir / "summary.json"))

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
        log_file = run_info.run_dir / "run.log"
        if log_file.exists():
            logger.info(InferLogEvent.LOGS_SAVED, path=str(log_file))
