"""Domain service for LLM inference pipeline.

This module contains pure business logic for orchestrating the inference process.
It depends only on port abstractions, has no knowledge of infrastructure details
(APIs, file formats, databases), and can be tested in complete isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable, Any

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import (
    LLMProvider,
    ExampleReader,
    JudgementWriter,
    ResponseParser,
    PromptBuilder,
)
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
from llm_ensemble.libs.schemas.write_result import WriteResult


class InferenceService:
    """Domain service for coordinating LLM inference pipeline.

    Pure business logic that orchestrates reading examples, running inference,
    and writing judgements. Depends only on port abstractions, enabling complete
    independence from infrastructure concerns.

    Example:
        >>> reader = NdjsonExampleReader()
        >>> writer = NdjsonJudgementWriter(output_path)
        >>> provider = OpenRouterAdapter()
        >>> service = InferenceService(reader, writer, provider)
        >>> stats = service.run_inference(
        ...     input_path,
        ...     model_config,
        ...     "thomas-et-al-prompt",
        ...     limit=100
        ... )
        >>> print(f"Processed {stats['judgement_count']} judgements")
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

    def run_inference(
        self,
        input_path: Path,
        model_config: ModelConfig,
        run_info: InferRunInfo,
        run_dir: Path,
        limit: Optional[int] = None,
        on_request_start: Optional[Callable[[], None]] = None,
        on_response: Optional[Callable[[LLMJudgement], None]] = None,
        on_write_one: Optional[Callable[[WriteResult], None]] = None,
        on_write_complete: Optional[Callable[[WriteSummary], None]] = None,
    ) -> InferRunSummary:
        """Execute the inference pipeline with streaming and immediate persistence.

        Pure business logic that coordinates:
        1. Reading JudgingSample objects via ExampleReader port
        2. For each sample (streaming loop):
           a. Building prompt via PromptBuilder port → str (rendered prompt)
           b. Running inference via LLMProvider port → LLMResponse (raw response + provider warnings)
           c. Parsing response via ResponseParser port → LLMScore (parsed score + parser warnings)
           d. Creating LLMJudgement object (sample + prompt + response + score + run_info)
           e. Invoking callback for progress tracking
           f. Writing judgement immediately to disk (fault tolerance)
        3. Calculating statistics including warnings summary from all stages

        Args:
            input_path: Path to input examples
            model_config: Model configuration
            run_info: Immutable runtime context (created by orchestrator, attached to each judgement)
            run_dir: Run directory where output should be written (writer determines file structure)
            limit: Optional maximum number of examples to process
            on_request_start: Optional callback invoked before sending request
            on_response: Optional callback for each completed judgement (for logging/progress)
            on_write_one: Optional callback invoked after each judgement is written (per-write logging)
            on_write_complete: Optional callback invoked after all writes complete (aggregate logging)

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

                # Invoke callback before sending request
                if on_request_start:
                    on_request_start()

                # Run inference for this sample 
                response = self.llm_provider.infer(prompt, model_config)

                # Parse response to extract structured score
                score = self.response_parser.parse(response.raw_response)

                # Create judgement immediately 
                judgement = LLMJudgement(
                    judging_sample=sample,
                    prompt=prompt,
                    llm_response=response,
                    llm_score=score,
                )

                # Invoke callback for progress tracking
                if on_response:
                    on_response(judgement)

                # Write judgement immediately to disk (fault tolerance!)
                write_result = writer.write_one(judgement)

                # Invoke callback after write (per-write logging)
                if on_write_one:
                    on_write_one(write_result)

                # Collect for summary statistics
                llm_judgements.append(judgement)

        # Retrieve aggregate write summary after context manager closes
        write_summary = self.judgement_writer.get_summary()

        # Invoke callback after all writes complete
        if on_write_complete:
            on_write_complete(write_summary)

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
