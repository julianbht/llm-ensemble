"""Domain service for data ingestion pipeline.

This module contains pure business logic for orchestrating the ingestion process.
It depends only on port abstractions, has no knowledge of infrastructure details
(file formats, I/O operations), and can be tested in complete isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary, NormalizedDataset
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.ports import DatasetReader, DatasetWriter
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder


class IngestionService:
    """Domain service for coordinating data ingestion pipeline.

    Pure business logic that orchestrates reading raw datasets and writing output.
    Depends only on port abstractions, enabling complete independence from
    infrastructure concerns.
    """

    def __init__(
        self,
        dataset_reader: DatasetReader,
        dataset_writer: DatasetWriter,
    ):
        """Initialize ingestion service with port dependencies.

        Args:
            dataset_reader: Port for reading and normalizing datasets
            dataset_writer: Port for writing judging samples
        """
        self.dataset_reader = dataset_reader
        self.dataset_writer = dataset_writer

    def ingest_dataset(
        self,
        data_dir: Path,
        run_info: IngestRunInfo,
        limit: Optional[int] = None,
        on_sample: Optional[Callable[[JudgingSample], None]] = None,
        on_write: Optional[Callable[[WriteSummary], None]] = None,
    ) -> IngestRunSummary:
        """Execute the ingestion pipeline.

        Pure business logic that coordinates:
        1. Creating run summary builder with timing
        2. Reading complete NormalizedDataset via DatasetReader port (reader handles normalization)
        3. Writing samples via DatasetWriter port (writer derives output path from run_info)
        4. Calculating summary statistics and finalizing

        Args:
            data_dir: Directory containing raw dataset files
            run_info: Immutable runtime context (contains run_dir property for writers)
            limit: Optional maximum number of samples to process
            on_sample: Optional callback invoked for each sample (for logging/progress)
            on_write: Optional callback invoked after batch write completes (for logging)

        Returns:
            Finalized IngestRunSummary with sample_count and timing information

        Raises:
            FileNotFoundError: If dataset files are missing
            ValueError: If dataset files are malformed
            Exception: If any step in the pipeline fails
        """
        # Create run summary builder (for timing and collection of metrics)
        summary_builder = RunSummaryBuilder()
        summary_builder.set_start_time()

        # Read and normalize dataset (reader creates complete JudgingSamples with UUIDs)
        normalized_dataset: NormalizedDataset = self.dataset_reader.read(
            data_dir,
            limit=limit
        )

        sample_count = normalized_dataset.sample_count

        # Invoke callback for each sample if provided (for logging/progress tracking)
        if on_sample:
            for sample in normalized_dataset.samples:
                on_sample(sample)

        # Write normalized dataset (writer derives output location from run_info)
        write_summary = self.dataset_writer.write(
            normalized_dataset,
            run_info
        )

        # Invoke callback after write (for logging)
        if on_write:
            on_write(write_summary)

        # Add write summary to builder for inclusion in final summary
        summary_builder.add("write_summary", write_summary)

        # Add statistics to summary builder
        summary_builder.add("sample_count", sample_count)

        # Finalize summary (sets end_time and creates immutable Pydantic object)
        summary: IngestRunSummary = summary_builder.finalize(IngestRunSummary)

        # Return finalized summary
        return summary
