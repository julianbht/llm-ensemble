"""Domain service for data ingestion pipeline.

This module contains pure business logic for orchestrating the ingestion process.
It depends only on port abstractions, has no knowledge of infrastructure details
(file formats, I/O operations), and can be tested in complete isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.ingest.ports import DatasetAdapter, ExampleWriter


class IngestionService:
    """Domain service for coordinating data ingestion pipeline.

    Pure business logic that orchestrates reading raw datasets, normalizing them
    into JudgingExamples, and writing output. Depends only on port abstractions,
    enabling complete independence from infrastructure concerns.

    Example:
        >>> adapter = LlmJudgeAdapter()
        >>> writer = NdjsonExampleWriter(output_path)
        >>> service = IngestionService(adapter, writer)
        >>> stats = service.run_ingestion(data_dir, limit=100)
        >>> print(f"Processed {stats['sample_count']} examples")
    """

    def __init__(
        self,
        dataset_adapter: DatasetAdapter,
        example_writer: ExampleWriter,
    ):
        """Initialize ingestion service with port dependencies.

        Args:
            dataset_adapter: Port for reading raw datasets
            example_writer: Port for writing normalized examples
        """
        self.dataset_adapter = dataset_adapter
        self.example_writer = example_writer

    def run_ingestion(
        self,
        data_dir: Path,
        limit: Optional[int] = None,
        on_example: Optional[Callable[[JudgingExample], None]] = None,
    ) -> dict:
        """Execute the ingestion pipeline.

        Pure business logic that coordinates:
        1. Reading raw dataset via DatasetAdapter port
        2. Writing normalized examples via ExampleWriter port
        3. Collecting statistics

        Args:
            data_dir: Directory containing raw dataset files
            limit: Optional maximum number of examples to process
            on_example: Optional callback invoked for each example (for logging/progress)

        Returns:
            Dictionary with statistics:
            - sample_count: Total number of examples processed

        Raises:
            FileNotFoundError: If dataset files are missing
            ValueError: If dataset files are malformed
            Exception: If any step in the pipeline fails
        """
        # Track statistics
        count = 0

        # Run ingestion pipeline
        for example in self.dataset_adapter.read(data_dir):
            # Write example
            self.example_writer.write(example)

            # Update statistics
            count += 1

            # Invoke callback if provided (for logging/progress tracking)
            if on_example:
                on_example(example)

            # Stop if limit reached
            if limit is not None and count >= limit:
                break

        # Finalize writer
        self.example_writer.close()

        # Return statistics
        return {
            "sample_count": count,
        }
