"""Domain service for LLM inference pipeline.

This module contains business logic for coordinating the inference process.
It depends only on port abstractions and handles its own logging.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import structlog

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement, LLMResponse, LLMScore
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
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        response_parser: ResponseParser,
    ):
        """Initialize inference service with port dependencies.

        Args:
            example_reader: Port for reading judging examples
            judgement_writer: Port for writing model judgements
            prompt_builder: Port for building prompts from samples
            llm_provider: Port for LLM inference (accepts prompts, returns raw responses)
            response_parser: Port for parsing raw responses into structured scores
        """
        self.example_reader = example_reader
        self.judgement_writer = judgement_writer
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.response_parser = response_parser
        self.logger = structlog.get_logger().bind(component="inference_service")

    def run_inference(
        self,
        input_path: Path,
        model_config: ModelConfig,
        run_info: InferRunInfo,
        run_dir: Path,
        limit: Optional[int] = None,
    ) -> InferRunSummary:
        """Execute the inference pipeline with streaming and immediate persistence.

        Coordinates:
        1. Reading JudgingSample objects via ExampleReader port
        2. For each sample (streaming loop):
           a. Building prompt via PromptBuilder port → str (rendered prompt)
           b. Running inference via LLMProvider port → LLMResponse (raw response + provider warnings)
           c. Parsing response via ResponseParser port → LLMScore (parsed score + parser warnings)
           d. Creating LLMJudgement object (sample + prompt + response + score + run_info)
           e. Logging progress
           f. Writing judgement immediately to disk (fault tolerance)
        3. Calculating statistics including warnings summary from all stages

        Args:
            input_path: Path to input examples
            model_config: Model configuration
            run_info: Immutable runtime context (created by orchestrator, attached to each judgement)
            run_dir: Run directory where output should be written (writer determines file structure)
            limit: Optional maximum number of examples to process

        Returns:
            Finalized InferRunSummary with statistics, timing, and warnings summary

        Raises:
            Exception: If any step in the pipeline fails
        """
        summary_builder = RunSummaryBuilder()
        summary_builder.set_start_time()

        # Read JudgingSample objects
        samples = self.example_reader.read(input_path, limit=limit)

        # Collect judgements for summary statistics
        llm_judgements: list[LLMJudgement] = []

        # Open writer for streaming (context manager ensures proper cleanup)
        with self.judgement_writer.open(run_dir, run_info) as writer:
            # Process each sample individually (streaming loop)
            for sample in samples:
                # Build prompt for this sample
                prompt = self.prompt_builder.build(sample)

                # Log before sending request
                self.logger.info(InferLogEvent.SENDING_REQUEST)

                # Run inference for this sample
                response : LLMResponse = self.llm_provider.infer(prompt, model_config)

                # Parse response to extract structured score
                score : LLMScore = self.response_parser.parse(response.raw_response)

                # Create judgement
                judgement = LLMJudgement(
                    judging_sample=sample,
                    prompt=prompt,
                    llm_response=response,
                    llm_score=score,
                )

                # Log each completed judgement
                extracted_score = judgement.llm_score.label.value if judgement.llm_score.label else "null"
                gold_score = judgement.judging_sample.gold_score.value
                latency_s = judgement.llm_response.latency_ms / 1000

                # Info to console
                self.logger.info(
                    InferLogEvent.RESPONSE_PARSED,
                    extracted_score=extracted_score,
                    gold_score=gold_score,
                    latency_s=f"{latency_s:.1f}",
                )

                # Full details (DEBUG level)
                self.logger.debug(
                    "judgement_details",
                    query=judgement.judging_sample.query.query_text,
                    doc=judgement.judging_sample.document.doc_text,
                    prompt=judgement.prompt,
                    raw_response=judgement.llm_response.raw_response,
                    extracted_score=extracted_score,
                    gold_score=gold_score,
                    latency_ms=judgement.llm_response.latency_ms,
                    warnings=[str(w) for w in judgement.get_all_warnings()],
                )

                # Write judgement immediately to disk (fault tolerance!)
                # Adapter handles logging of what entities were persisted
                writer.write_one(judgement)

                # Collect for summary statistics
                llm_judgements.append(judgement)

        # Retrieve aggregate write summary after context manager closes (writer logged directly)
        write_summary = self.judgement_writer.get_summary()

        # Calculate aggregate statistics from judgements (for summary)
        count = len(llm_judgements)
        error_count = sum(1 for j in llm_judgements if j.llm_score.label is None)
        total_latency_ms = sum(j.llm_response.latency_ms for j in llm_judgements)
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
