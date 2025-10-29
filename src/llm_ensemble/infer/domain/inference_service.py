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
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig, InferManifest
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
        limit: Optional[int] = None,
        on_judgement: Optional[Callable[[ModelJudgement], None]] = None,
    ) -> InferManifest:
        """Execute the inference pipeline.

        Pure business logic that coordinates:
        1. Setting start_time in the manifest builder
        2. Reading examples via ExampleReader port
        3. Running inference via LLMProvider port
        4. Writing judgements via JudgementWriter port
        5. Collecting statistics
        6. Adding statistics to manifest builder
        7. Finalizing the manifest (sets end_time)

        Args:
            input_path: Path to input examples
            model_config: Model configuration
            manifest_builder: Manifest builder for constructing final manifest
            limit: Optional maximum number of examples to process
            on_judgement: Optional callback invoked for each judgement (for logging/progress)

        Returns:
            Finalized InferManifest with statistics and timing information

        Raises:
            Exception: If any step in the pipeline fails
        """
        # Set start_time when processing begins
        manifest_builder.add("start_time", datetime.now())

        # Read examples
        examples = self.example_reader.read(input_path, limit=limit)

        # Track statistics
        count = 0
        error_count = 0
        total_latency_ms = 0.0

        # Run inference pipeline
        for judgement in self._process_examples(examples, model_config):
            # Write judgement
            self.judgement_writer.write(judgement)

            # Update statistics
            count += 1
            total_latency_ms += judgement.latency_ms
            if judgement.label is None:
                error_count += 1

            # Invoke callback if provided (for logging/progress tracking)
            if on_judgement:
                on_judgement(judgement)

        # Finalize writer
        self.judgement_writer.close()

        # Calculate average latency
        avg_latency = total_latency_ms / count if count > 0 else 0.0

        # Add statistics to manifest builder
        manifest_builder.add("judgement_count", count)
        manifest_builder.add("error_count", error_count)
        manifest_builder.add("total_latency_ms", total_latency_ms)
        manifest_builder.add("avg_latency_ms", avg_latency)

        # Finalize manifest (sets end_time and creates immutable Pydantic object)
        manifest: InferManifest = manifest_builder.finalize(InferManifest)

        # Return finalized manifest
        return manifest

    def _process_examples(
        self,
        examples: list[JudgingSample],
        model_config: ModelConfig,
    ) -> Iterator[ModelJudgement]:
        """Process examples through LLM provider.

        Args:
            examples: List of judging examples
            model_config: Model configuration

        Yields:
            ModelJudgement objects from inference
        """
        yield from self.llm_provider.infer(iter(examples), model_config)
