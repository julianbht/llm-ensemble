"""Domain service for LLM inference pipeline.

This module contains pure business logic for orchestrating the inference process.
It depends only on port abstractions, has no knowledge of infrastructure details
(APIs, file formats, databases), and can be tested in complete isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterator, Callable

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider, ExampleReader, JudgementWriter


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
        limit: Optional[int] = None,
        on_judgement: Optional[Callable[[ModelJudgement], None]] = None,
    ) -> dict:
        """Execute the inference pipeline.

        Pure business logic that coordinates:
        1. Reading examples via ExampleReader port
        2. Running inference via LLMProvider port
        3. Writing judgements via JudgementWriter port
        4. Collecting statistics

        Args:
            input_path: Path to input examples
            model_config: Model configuration
            limit: Optional maximum number of examples to process
            on_judgement: Optional callback invoked for each judgement (for logging/progress)

        Returns:
            Dictionary with statistics:
            - judgement_count: Total judgements processed
            - error_count: Number of failed judgements (label=None)
            - total_latency_ms: Total latency across all judgements
            - avg_latency_ms: Average latency per judgement

        Raises:
            Exception: If any step in the pipeline fails
        """
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

        # Calculate and return statistics
        avg_latency = total_latency_ms / count if count > 0 else 0.0

        return {
            "judgement_count": count,
            "error_count": error_count,
            "total_latency_ms": total_latency_ms,
            "avg_latency_ms": avg_latency,
        }

    def _process_examples(
        self,
        examples: list[JudgingExample],
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
