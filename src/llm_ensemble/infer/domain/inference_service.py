"""Domain service for LLM inference pipeline.

This module contains business logic for coordinating the inference process.
It depends only on port abstractions and handles its own logging.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement, LLMScore
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary
from llm_ensemble.infer.ports import (
    LLMProvider,
    ExampleReader,
    JudgementWriter,
    ResponseParser,
    PromptBuilder,
)
from llm_ensemble.libs.registry import AdapterWithMetadata
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
        example_reader: ExampleReader,
        judgement_writer: JudgementWriter,
        prompt_adapter: AdapterWithMetadata,
        llm_provider: LLMProvider,
        parser_adapter: AdapterWithMetadata,
    ):
        """Initialize inference service with port dependencies.

        Args:
            example_reader: Port for reading judging examples
            judgement_writer: Port for writing model judgements
            prompt_adapter: Prompt builder wrapped with metadata (adapter + name)
            llm_provider: Port for LLM inference (accepts prompts, returns raw responses)
            parser_adapter: Response parser wrapped with metadata (adapter + name)
        """
        self.example_reader = example_reader
        self.judgement_writer = judgement_writer
        self.prompt_adapter = prompt_adapter
        self.llm_provider = llm_provider
        self.parser_adapter = parser_adapter
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
        normalized_dataset = self.example_reader.read(run_name)

        # Compute actual start_idx and end_idx from run_info
        start_idx = run_info.start_idx if run_info.start_idx is not None else 0
        end_idx = run_info.end_idx if run_info.end_idx is not None else len(normalized_dataset.samples)

        # Slice samples based on computed indices
        samples_to_process = normalized_dataset.samples[start_idx:end_idx]

        # Collect judgements for summary statistics
        llm_judgements: list[LLMJudgement] = []

        # Extract adapters from wrappers
        prompt_builder = self.prompt_adapter.adapter
        response_parser = self.parser_adapter.adapter

        # Compute UUIDs from identity (names from registry)
        from llm_ensemble.libs.db import compute_prompt_template_uuid, compute_parser_spec_uuid_from_name
        prompt_template_id = compute_prompt_template_uuid(self.prompt_adapter.name)
        parser_spec_id = compute_parser_spec_uuid_from_name(self.parser_adapter.name)

        # Open writer for streaming (context manager ensures proper cleanup)
        # Pass computed indices to writer so it can create InferRun entity with actual range
        with self.judgement_writer.open(run_dir, run_info, normalized_dataset, start_idx, end_idx) as writer:
            # Process each dataset sample in slice (streaming loop)
            for dataset_sample in samples_to_process:
                # Build prompt from dataset_sample (adapter returns raw tuple)
                ds, prompt_text = prompt_builder.build_raw(dataset_sample)

                # Create LLMPrompt domain object with identity from metadata
                from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt
                llm_prompt = LLMPrompt.create(
                    dataset_sample=ds,
                    prompt_text=prompt_text,
                    prompt_template_id=prompt_template_id,
                )

                # Run inference - returns raw text and metrics
                self.logger.info(InferLogEvent.SENDING_REQUEST)
                raw_response_text, invocation_metrics = self.llm_provider.infer(
                    llm_prompt.prompt_text,
                    model_config
                )

                # Parse response to extract structured score (adapter returns DTO)
                from llm_ensemble.infer.schemas.llm_judgement import LLMScore
                parsed_dto = response_parser.parse_raw(raw_response_text)

                # Create LLMScore domain object with identity from metadata
                llm_score = LLMScore.create(
                    llm_response_text=parsed_dto.llm_response_text,
                    parser_spec_id=parser_spec_id,
                    label=parsed_dto.label,
                    confidence=parsed_dto.confidence,
                    rationale=parsed_dto.rationale,
                    warnings=parsed_dto.warnings,
                )

                # Create judgement from nested components
                judgement = LLMJudgement.create(
                    llm_prompt=llm_prompt,
                    invocation_metrics=invocation_metrics,
                    llm_score=llm_score,
                )

                # Log each completed judgement
                extracted_score = judgement.llm_score.label.value if judgement.llm_score and judgement.llm_score.label else "null"
                gold_score = judgement.llm_prompt.dataset_sample.judging_sample.gold_score.value
                latency_s = judgement.invocation_metrics.latency_ms / 1000
                self.logger.info(
                    InferLogEvent.RESPONSE_PARSED,
                    extracted_score=extracted_score,
                    gold_score=gold_score,
                    latency_s=f"{latency_s:.1f}",
                )

                # Log metrics for observability dashboard (cost, agreement, latency)
                cost_usd = judgement.invocation_metrics.cost_estimate_usd or 0.0
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
        write_summary = self.judgement_writer.get_summary()

        # Calculate aggregate statistics from judgements (for summary)
        count = len(llm_judgements)
        error_count = sum(1 for j in llm_judgements if j.llm_score is None or j.llm_score.label is None)
        total_latency_ms = sum(j.invocation_metrics.latency_ms for j in llm_judgements)
        avg_latency = total_latency_ms / count if count > 0 else 0.0

        # Build warnings summary from all judgements
        warnings_summary: dict[str, int] = {}
        for judgement in llm_judgements:
            # Aggregate warnings from all stages (request + response + score)
            for warning in judgement.get_all_warnings():
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
