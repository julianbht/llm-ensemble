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
from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas import ModelConfig, InferManifest
from llm_ensemble.infer.ports import LLMProvider, ExampleReader, JudgementWriter
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
        llm_provider: LLMProvider,
    ):
        """Initialize inference service with port dependencies.

        Args:
            example_reader: Port for reading judging examples
            judgement_writer: Port for writing model judgements
            llm_provider: Port for LLM inference
        """
        self.example_reader = example_reader
        self.judgement_writer = judgement_writer
        self.llm_provider = llm_provider

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
        3. Running inference via LLMProvider port to get LLMResponse objects (sample + LLM output pairs)
        4. Calculating statistics and adding to manifest builder
        5. Finalizing the manifest (sets end_time)
        6. Attaching manifest to each (sample, LLMResponse) pair to create LLMJudgement objects
        7. Writing LLMJudgement objects via JudgementWriter port

        Args:
            input_path: Path to input examples
            model_config: Model configuration
            manifest_builder: Manifest builder for constructing final manifest
            run_dir: Run directory where output should be written (writer determines file structure)
            limit: Optional maximum number of examples to process
            on_judgement: Optional callback invoked for each judgement (for logging/progress)

        Returns:
            Finalized InferManifest with statistics and timing information

        Raises:
            Exception: If any step in the pipeline fails
        """
        # Set start_time when processing begins
        manifest_builder.add("start_time", datetime.now())

        # Read JudgingSample objects (which include ingest manifest)
        samples = self.example_reader.read(input_path, limit=limit)

        # Run inference to get (sample, LLMResponse) pairs
        sample_response_pairs = list(self._process_samples(samples, model_config))

        # Calculate statistics from LLMResponses
        count = len(sample_response_pairs)
        error_count = sum(1 for _, resp in sample_response_pairs if resp.llm_score is None)
        total_latency_ms = sum(resp.latency_ms for _, resp in sample_response_pairs)
        avg_latency = total_latency_ms / count if count > 0 else 0.0

        # Add statistics to manifest builder
        manifest_builder.add("judgement_count", count)
        manifest_builder.add("error_count", error_count)
        manifest_builder.add("total_latency_ms", total_latency_ms)
        manifest_builder.add("avg_latency_ms", avg_latency)

        # Finalize manifest (sets end_time and creates immutable Pydantic object)
        manifest: InferManifest = manifest_builder.finalize(InferManifest)

        # Attach manifest to each (sample, LLMResponse) pair to create LLMJudgement objects
        llm_judgements = [
            LLMJudgement(
                sample=sample,
                llm_response=response,
                manifest=manifest,
            )
            for sample, response in sample_response_pairs
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
        """Process samples through LLM provider to get responses.

        Args:
            samples: List of judging samples to process
            model_config: Model configuration

        Yields:
            Tuples of (sample, LLMResponse) for each inference
        """
        # Provider returns LLMResponse objects - we pair them with their input samples
        yield from self.llm_provider.infer(iter(samples), model_config)
