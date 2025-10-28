"""Domain service for data ingestion pipeline.

This module contains pure business logic for orchestrating the ingestion process.
It depends only on port abstractions, has no knowledge of infrastructure details
(file formats, I/O operations), and can be tested in complete isolation.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from llm_ensemble.ingest.schemas import JudgingSample, IngestManifest
from llm_ensemble.ingest.ports import SampleReader, DatasetWriter
from llm_ensemble.ingest.ports.sample_reader import RawJudgingSample
from llm_ensemble.libs.runtime.manifest_manager import ManifestBuilder


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
        manifest_builder: ManifestBuilder,
        run_dir: Path,
        limit: Optional[int] = None,
        on_sample: Optional[Callable[[JudgingSample], None]] = None,
    ) -> IngestManifest:
        """Execute the ingestion pipeline.

        Pure business logic that coordinates:
        1. Setting start_time in the manifest builder
        2. Reading RawJudgingSample DTOs from raw dataset via SampleReader port
        3. Adding sample_count to the manifest builder
        4. Finalizing the manifest (sets end_time)
        5. Attaching manifest to each sample (Many-to-One relationship) to create JudgingSamples
        6. Writing JudgingSamples via DatasetWriter port (writer determines output structure)

        Args:
            data_dir: Directory containing raw dataset files
            manifest_builder: Manifest builder for constructing final manifest
            run_dir: Run directory where output should be written (writer determines file structure)
            limit: Optional maximum number of samples to process
            on_sample: Optional callback invoked for each sample (for logging/progress)

        Returns:
            Finalized IngestManifest with sample_count and timing information

        Raises:
            FileNotFoundError: If dataset files are missing
            ValueError: If dataset files are malformed
            Exception: If any step in the pipeline fails
        """
        # Set start_time when processing begins
        manifest_builder.add("start_time", datetime.now())

        # Read RawJudgingSample DTOs from raw dataset (SampleReader handles limit internally)
        raw_samples: list[RawJudgingSample] = self.sample_reader.read(data_dir, limit=limit)

        sample_count = len(raw_samples)

        # Add sample_count to builder
        manifest_builder.add("sample_count", sample_count)

        # Finalize manifest (sets end_time and creates immutable Pydantic object)
        manifest: IngestManifest = manifest_builder.finalize(IngestManifest)

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

        # Invoke callback for each sample if provided (for logging/progress tracking)
        if on_sample:
            for sample in judging_samples:
                on_sample(sample)

        # Write samples (writer determines output file structure)
        self.dataset_writer.write(judging_samples, run_dir)

        # Return finalized manifest
        return manifest
