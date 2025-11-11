"""Domain service for data ingestion pipeline.

This module contains pure business logic for orchestrating the ingestion process.
It depends only on port abstractions, has no knowledge of infrastructure details
(file formats, I/O operations), and can be tested in complete isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.ports import SampleReader, DatasetWriter
from llm_ensemble.ingest.ports.sample_reader import RawJudgingSample
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder


class IngestionService:
    """Domain service for coordinating data ingestion pipeline.

    Pure business logic that orchestrates reading raw datasets and writing output.
    Depends only on port abstractions, enabling complete independence from
    infrastructure concerns.
    """

    def __init__(
        self,
        sample_reader: SampleReader,
        dataset_writer: DatasetWriter,
    ):
        """Initialize ingestion service with port dependencies.

        Args:
            sample_reader: Port for reading raw datasets
            dataset_writer: Port for writing judging samples
        """
        self.sample_reader = sample_reader
        self.dataset_writer = dataset_writer

    def ingest_dataset(
        self,
        data_dir: Path,
        run_info: IngestRunInfo,
        run_dir: Path,
        limit: Optional[int] = None,
        on_sample: Optional[Callable[[JudgingSample], None]] = None,
        on_write: Optional[Callable[[WriteSummary], None]] = None,
    ) -> IngestRunSummary:
        """Execute the ingestion pipeline.

        Pure business logic that coordinates:
        1. Creating run summary builder with timing
        2. Reading RawJudgingSample DTOs from raw dataset via SampleReader port
        3. Converting RawJudgingSamples to JudgingSamples (pure domain entities)
        4. Writing JudgingSamples via DatasetWriter port (writer determines output structure)
        5. Calculating summary statistics and finalizing

        Args:
            data_dir: Directory containing raw dataset files
            run_info: Immutable runtime context (passed to writer for persistence, not attached to samples)
            run_dir: Run directory where output should be written (writer determines file structure)
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

        # Extract dataset name and description from IngestIOConfig
        dataset_name = run_info.io_config.dataset_name
        dataset_description = run_info.io_config.dataset_description

        # Read RawJudgingSample DTOs from raw dataset (SampleReader handles limit internally)
        raw_samples: list[RawJudgingSample] = self.sample_reader.read(
            data_dir,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            limit=limit
        )

        sample_count = len(raw_samples)

        # Convert RawJudgingSamples to JudgingSamples (pure domain entities)
        judging_samples = [
            JudgingSample.create(
                query=sample.query,
                document=sample.document,
                gold_score=sample.gold_score,
            )
            for sample in raw_samples
        ]

        # Invoke callback for each sample if provided (for logging/progress tracking)
        if on_sample:
            for sample in judging_samples:
                on_sample(sample)

        # Write samples 
        write_summary = self.dataset_writer.write(judging_samples, run_dir, run_info)

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
