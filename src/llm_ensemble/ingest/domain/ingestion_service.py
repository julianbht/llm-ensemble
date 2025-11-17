"""Domain service for data ingestion pipeline.

This module contains business logic for coordinating the ingestion process.
It depends only on port abstractions and handles its own logging.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import NormalizedDataset
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.ports import DatasetReader, DatasetWriter
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
from llm_ensemble.libs.logging.log_events import IngestLogEvent


class IngestionService:
    """Domain service for coordinating data ingestion pipeline.

    Business logic that orchestrates reading raw datasets and writing output.
    Handles its own logging - no callback injection needed.
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
        self.logger = get_logger(component="ingestion_service")

    def ingest_dataset(
        self,
        data_dir: Path,
        run_info: IngestRunInfo,
        limit: Optional[int] = None,
    ) -> IngestRunSummary:
        """Execute the ingestion pipeline.

        Coordinates:
        1. Creating run summary builder with timing
        2. Reading complete NormalizedDataset via DatasetReader port
        3. Writing samples via DatasetWriter port
        4. Calculating summary statistics and finalizing

        Args:
            data_dir: Directory containing raw dataset files
            run_info: Immutable runtime context (contains run_dir property for writers)
            limit: Optional maximum number of samples to process

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

        # Read and normalize dataset
        normalized_dataset: NormalizedDataset = self.dataset_reader.read(
            data_dir,
            limit=limit
        )

        # Log read completion
        self.logger.info(
            IngestLogEvent.DATASET_READ_COMPLETE,
            dataset=normalized_dataset.dataset.name,
            sample_count=normalized_dataset.sample_count,
        )

        # Write normalized dataset (writer logs directly)
        write_summary = self.dataset_writer.write(
            normalized_dataset,
            run_info
        )

        # Add to write summary to builder for inclusion in final summary
        summary_builder.add("write_summary", write_summary)
        summary_builder.add("sample_count", normalized_dataset.sample_count)

        # Finalize summary (sets end_time and creates immutable Pydantic object)
        summary: IngestRunSummary = summary_builder.finalize(IngestRunSummary)

        # Return finalized summary
        return summary
