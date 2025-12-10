"""Domain service for LLM inference pipeline.

This module contains business logic for coordinating the inference process.
It depends only on port abstractions and handles its own logging.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary
from llm_ensemble.infer.ports import (
    LLMProviderPort,
    InputPort,
    OutputPort,
    ResponseParserPort,
    PromptBuilderPort,
)
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
from llm_ensemble.libs.logging.log_events import InferLogEvent


class InferenceService:
    """Domain service for coordinating LLM inference pipeline.

    Business logic that orchestrates reading examples, running inference,
    and writing judgements. Handles its own logging - no callback injection needed.
    """

    def __init__(
        self,
        input_adapter: InputPort,
        output_adapter: OutputPort,
        prompt_builder: PromptBuilderPort,
        llm_provider: LLMProviderPort,
        response_parser: ResponseParserPort,
    ):
        """Initialize inference service with port dependencies.

        Args:
            input_adapter: Port for reading input data
            output_adapter: Port for writing output data
            prompt_builder: Port for building prompts from samples
            llm_provider: Port for LLM inference
            response_parser: Port for parsing LLM responses
        """
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.response_parser = response_parser
        self.logger = get_logger(component="inference_service")

    def run_inference(
        self,
        run_name: str,
        model_config: ModelConfig,
        run_info: InferRunInfo,
        run_dir: Path,
    ) -> InferRunSummary:
        """Execute the inference pipeline with streaming and immediate persistence.

        Coordinates:
        1. Reading DatasetSample objects via ExampleReader port
        2. Computing actual start_idx and end_idx from run_info (defaulting to 0 and sample_count)
        3. Slicing samples based on computed indices
        4. For each dataset_sample in slice (streaming loop):
           a. Building prompt via PromptBuilder port → LLMPrompt (dataset_sample + prompt_text)
           b. Running inference via LLMProvider port → (raw_response_text, LLMInvocationMetrics)
           c. Parsing response via ResponseParser port → LLMScore (llm_response_text + parsed fields)
           d. Creating LLMJudgement object (llm_prompt + invocation_metrics + llm_score)
           e. Logging progress
           f. Writing judgement immediately to disk (fault tolerance)
        5. Calculating statistics including warnings summary from all stages

        Args:
            run_name: Ingest run identifier (reader resolves to file path)
            model_config: Model configuration
            run_info: Immutable runtime context with nullable start_idx/end_idx capturing user intent
            run_dir: Run directory where output should be written (writer determines file structure)

        Returns:
            Finalized InferRunSummary with statistics, timing, and warnings summary

        Raises:
            Exception: If any step in the pipeline fails
        """
        summary_builder = RunSummaryBuilder()
        summary_builder.set_start_time()

        # Read full NormalizedDataset
        normalized_dataset = self.input_adapter.read(run_name)

        # Compute actual start_idx and end_idx from run_info
        start_idx = run_info.start_idx if run_info.start_idx is not None else 0
        end_idx = run_info.end_idx if run_info.end_idx is not None else len(normalized_dataset.samples)

        # Slice samples based on computed indices
        samples_to_process = normalized_dataset.samples[start_idx:end_idx]

        # Collect judgements for summary statistics
        llm_judgements: list[LLMJudgement] = []

        # Open writer for streaming
        with self.output_adapter.open(run_dir, run_info, normalized_dataset) as writer:

            # Process each dataset sample in slice (streaming loop)
            for dataset_sample in samples_to_process:

                # Build prompt text
                prompt_text = self.prompt_builder.build_prompt(dataset_sample)

                # Run inference
                self.logger.info(InferLogEvent.SENDING_REQUEST)
                raw_response_text, llm_invocation_metrics = self.llm_provider.infer(
                    prompt_text,
                    model_config
                )

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
                self.logger.info(
                    InferLogEvent.RESPONSE_PARSED,
                    extracted_score=extracted_score,
                    gold_score=gold_score,
                    latency_s=f"{latency_s:.1f}",
                )

                # Log metrics for observability dashboard (cost, agreement, latency)
                cost_usd = judgement.llm_invocation_metrics.cost_estimate_usd or 0.0
                agreement = 1 if extracted_score == gold_score else 0
                self.logger.info(
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
        write_summary = self.output_adapter.get_summary()

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

        # Return finalized summary
        return summary
