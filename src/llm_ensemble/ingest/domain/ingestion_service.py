"""Domain service for data ingestion pipeline.

This module contains pure business logic for orchestrating the ingestion process.
It depends only on port abstractions, has no knowledge of infrastructure details
(file formats, I/O operations), and can be tested in complete isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.ports import SampleReader, DatasetWriter
from llm_ensemble.ingest.ports.sample_reader import RawJudgingSample
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder


class IngestionService:
    """Domain service for coordinating data ingestion pipeline.

    Pure business logic that orchestrates reading raw datasets, attaching
    manifest to each sample (Many-to-One relationship), and writing output.
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
            dataset_writer: Port for writing judging samples with manifest
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
    ) -> IngestRunSummary:
        """Execute the ingestion pipeline.

        Pure business logic that coordinates:
        1. Creating run summary builder with timing
        2. Reading RawJudgingSample DTOs from raw dataset via SampleReader port
        3. Attaching run_info to each sample (Many-to-One relationship) to create JudgingSamples
        4. Writing JudgingSamples via DatasetWriter port (writer determines output structure)
        5. Calculating summary statistics and finalizing

        Args:
            data_dir: Directory containing raw dataset files
            run_info: Immutable runtime context (created by orchestrator, attached to each sample)
            run_dir: Run directory where output should be written (writer determines file structure)
            limit: Optional maximum number of samples to process
            on_sample: Optional callback invoked for each sample (for logging/progress)

        Returns:
            Finalized IngestRunSummary with sample_count and timing information

        Raises:
            FileNotFoundError: If dataset files are missing
            ValueError: If dataset files are malformed
            Exception: If any step in the pipeline fails
        """
        # Create run summary builder (for timing and collection of metrics)
        summary_builder = RunSummaryBuilder(run_info)
        summary_builder.set_start_time()

        # Extract dataset name and description from IngestIOConfig
        dataset_name = run_info.io_config.dataset_name
        dataset_description = run_info.io_config.dataset_description

        # Read RawJudgingSample DTOs from raw dataset (SampleReader handles limit internally)
        # Pass dataset_name and dataset_description so reader can create Dataset entity
        raw_samples: list[RawJudgingSample] = self.sample_reader.read(
            data_dir, 
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            limit=limit
        )

        sample_count = len(raw_samples)

        # Attach run_info to each sample (Many-to-One relationship)
        # Use JudgingSample.create() to compute deterministic UUID
        judging_samples = [
            JudgingSample.create(
                query=sample.query,
                document=sample.document,
                gold_score=sample.gold_score,
                run_info=run_info,
            )
            for sample in raw_samples
        ]

        # Invoke callback for each sample if provided (for logging/progress tracking)
        if on_sample:
            for sample in judging_samples:
                on_sample(sample)

        # Write samples (writer determines output file structure)
        self.dataset_writer.write(judging_samples, run_dir)

        # Add statistics to summary builder
        summary_builder.add("sample_count", sample_count)

        # Finalize summary (sets end_time and creates immutable Pydantic object)
        summary: IngestRunSummary = summary_builder.finalize(IngestRunSummary)

        # Return finalized summary
        return summary
