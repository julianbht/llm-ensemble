"""Domain service for LLM inference pipeline.

This module contains pure business logic for orchestrating the inference process.
It depends only on port abstractions, has no knowledge of infrastructure details
(APIs, file formats, databases), and can be tested in complete isolation.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Callable

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas.llm_score import LLMScore
from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas import ModelConfig, InferManifest
from llm_ensemble.infer.ports import (
    LLMProvider,
    ExampleReader,
    JudgementWriter,
    ResponseParser,
    PromptBuilder,
)
from llm_ensemble.libs.runtime.manifest_manager import ManifestBuilder


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
        manifest_builder: ManifestBuilder,
        run_dir: Path,
        limit: Optional[int] = None,
        on_judgement: Optional[Callable[[LLMJudgement], None]] = None,
    ) -> InferManifest:
        """Execute the inference pipeline.

        Pure business logic that coordinates:
        1. Setting start_time in the manifest builder
        2. Reading JudgingSample objects via ExampleReader port
        3. Building prompts via PromptBuilder port for each sample
        4. Running inference via LLMProvider port to get raw LLMResponse objects
        5. Parsing each LLMResponse via ResponseParser port to extract LLMScore
        6. Creating LLMJudgement objects (sample + raw response + parsed score)
        7. Calculating statistics including warnings summary from judgements
        8. Adding statistics to manifest builder
        9. Finalizing the manifest (sets end_time)
        10. Writing LLMJudgement objects via JudgementWriter port

        Args:
            input_path: Path to input examples
            model_config: Model configuration
            manifest_builder: Manifest builder for constructing final manifest
            run_dir: Run directory where output should be written (writer determines file structure)
            limit: Optional maximum number of examples to process
            on_judgement: Optional callback invoked for each judgement (for logging/progress)

        Returns:
            Finalized InferManifest with statistics, timing, and warnings summary

        Raises:
            Exception: If any step in the pipeline fails
        """
        # Set start_time when processing begins
        manifest_builder.add("start_time", datetime.now())

        # Read JudgingSample objects (which include ingest manifest)
        samples = self.example_reader.read(input_path, limit=limit)

        # Run inference to get (sample, LLMResponse) pairs (raw responses)
        sample_response_pairs = list(self._process_samples(samples, model_config))

        # Parse each raw response to extract structured scores
        sample_response_score_tuples = []
        for sample, raw_response in sample_response_pairs:
            # Parser now returns LLMScore with warnings included
            score = self.response_parser.parse(raw_response.raw_response)
            sample_response_score_tuples.append((sample, raw_response, score))

        # Calculate statistics from parsed scores
        count = len(sample_response_score_tuples)
        error_count = sum(1 for _, _, score in sample_response_score_tuples if score.label is None)
        total_latency_ms = sum(resp.latency_ms for _, resp, _ in sample_response_score_tuples)
        avg_latency = total_latency_ms / count if count > 0 else 0.0

        # Build warnings summary from responses and scores
        warnings_summary: dict[str, int] = {}
        for _, response, score in sample_response_score_tuples:
            # Count provider warnings from response
            for warning in response.warnings:
                warning_type = warning.__class__.__name__
                warnings_summary[warning_type] = warnings_summary.get(warning_type, 0) + 1
            # Count parser warnings from score
            for warning in score.warnings:
                warning_type = warning.__class__.__name__
                warnings_summary[warning_type] = warnings_summary.get(warning_type, 0) + 1

        # Add statistics to manifest builder
        manifest_builder.add("judgement_count", count)
        manifest_builder.add("error_count", error_count)
        manifest_builder.add("total_latency_ms", total_latency_ms)
        manifest_builder.add("avg_latency_ms", avg_latency)

        # Add warnings summary to manifest (only if there are warnings)
        if warnings_summary:
            manifest_builder.add("warnings_summary", warnings_summary)

        # Finalize manifest (sets end_time and creates immutable Pydantic object)
        manifest: InferManifest = manifest_builder.finalize(InferManifest)

        # Create LLMJudgement objects (sample + raw response + parsed score + manifest)
        llm_judgements = [
            LLMJudgement(
                sample=sample,
                llm_response=response,
                llm_score=score,
                manifest=manifest,
            )
            for sample, response, score in sample_response_score_tuples
        ]

        # Invoke callback for each judgement if provided (for logging/progress tracking)
        if on_judgement:
            for judgement in llm_judgements:
                on_judgement(judgement)

        # Write judgements (writer determines output file structure)
        self.judgement_writer.write(llm_judgements, run_dir)

        # Return finalized manifest
        return manifest

    def _process_samples(
        self,
        samples: list[JudgingSample],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Process samples through the full inference pipeline.

        Orchestrates:
        1. Building prompts for each sample via PromptBuilder
        2. Sending (sample, prompt) pairs to LLMProvider
        3. Receiving (sample, LLMResponse) pairs back

        Args:
            samples: List of judging samples to process
            model_config: Model configuration

        Yields:
            Tuples of (sample, LLMResponse) for each inference
        """
        # Step 1: Build prompts for each sample using PromptBuilder port
        sample_prompt_pairs = [
            (sample, self.prompt_builder.build(sample))
            for sample in samples
        ]

        # Step 2: Send (sample, prompt) pairs to provider and get (sample, response) back
        yield from self.llm_provider.infer(iter(sample_prompt_pairs), model_config)
