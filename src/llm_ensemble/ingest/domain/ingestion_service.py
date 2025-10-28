"""Domain service for data ingestion pipeline.

This module contains pure business logic for orchestrating the ingestion process.
It depends only on port abstractions, has no knowledge of infrastructure details
(file formats, I/O operations), and can be tested in complete isolation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable

from llm_ensemble.ingest.schemas import JudgingSample, IngestManifest
from llm_ensemble.ingest.ports import SampleReader, DatasetWriter
from llm_ensemble.ingest.ports.sample_reader import RawJudgingSample


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
        manifest: IngestManifest,
        output_path: Path,
        limit: Optional[int] = None,
        on_sample: Optional[Callable[[JudgingSample], None]] = None,
    ) -> dict:
        """Execute the ingestion pipeline.

        Pure business logic that coordinates:
        1. Reading RawJudgingSample DTOs from raw dataset via SampleReader port
        2. Attaching manifest to each sample (Many-to-One relationship) to create JudgingSamples
        3. Updating manifest with sample_count
        4. Writing JudgingSamples via DatasetWriter port
        5. Collecting statistics

        Args:
            data_dir: Directory containing raw dataset files
            manifest: Pre-built manifest (sample_count will be updated after reading)
            output_path: Path where dataset should be written
            limit: Optional maximum number of samples to process
            on_sample: Optional callback invoked for each sample (for logging/progress)

        Returns:
            Dictionary with statistics:
            - sample_count: Total number of samples processed

        Raises:
            FileNotFoundError: If dataset files are missing
            ValueError: If dataset files are malformed
            Exception: If any step in the pipeline fails
        """
        # Read RawJudgingSample DTOs from raw dataset (SampleReader handles limit internally)
        raw_samples: list[RawJudgingSample] = self.sample_reader.read(data_dir, limit=limit)

        # Attach manifest to each sample (Many-to-One relationship)
        judging_samples = [
            JudgingSample(
                query=sample.query,
                document=sample.document,
                gold_score=sample.gold_score,
                manifest=manifest,
            )
            for sample in raw_samples
        ]

        sample_count = len(judging_samples)

        # Invoke callback for each sample if provided (for logging/progress tracking)
        if on_sample:
            for sample in judging_samples:
                on_sample(sample)

        # Update manifest with actual sample count
        manifest.sample_count = sample_count

        # Write samples (each sample contains full manifest)
        self.dataset_writer.write(judging_samples, output_path)

        # Return statistics
        return {
            "sample_count": sample_count,
        }
