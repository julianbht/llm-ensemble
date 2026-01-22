"""Mock adapter implementations for testing the ingest pipeline.

This module provides test doubles (mocks) for all driven ports in the ingest CLI.
These mocks replace real infrastructure (file I/O, databases) with
in-memory implementations, enabling fast and deterministic testing of business logic.

Mock adapters are reusable across the ingest test suite.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.application.ports.driven.for_input import ForInput
from llm_ensemble.ingest.application.ports.driven.for_output import ForOutput
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run import IngestRun
from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary


class MockInputAdapter(ForInput):
    """Mock input adapter that returns predefined test data.

    Replaces real file/database readers with in-memory test data.
    Allows testing without file I/O dependencies.
    """

    def __init__(self, mock_dataset: NormalizedDataset):
        """Initialize with test dataset.

        Args:
            mock_dataset: Predefined dataset to return
        """
        self.mock_dataset = mock_dataset
        self.read_called = False
        self.read_call_args: dict[str, Optional[Path | int]] = {}

    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> NormalizedDataset:
        """Return mock dataset and track call."""
        self.read_called = True
        self.read_call_args = {"input_path": input_path, "limit": limit}

        # Apply limit if specified
        if limit is not None and limit < len(self.mock_dataset.samples):
            limited_dataset = NormalizedDataset(
                id=self.mock_dataset.id,
                fingerprint=self.mock_dataset.fingerprint,
                external_dataset_name=self.mock_dataset.external_dataset_name,
                samples=self.mock_dataset.samples[:limit],
            )
            return limited_dataset

        return self.mock_dataset


class MockOutputAdapter(ForOutput):
    """Mock output adapter that captures written data in memory.

    Replaces real file/database writers with in-memory collection.
    Allows verification of what was written without actual I/O.
    """

    def __init__(self):
        """Initialize empty collections for tracking writes."""
        self.written_ingest_run: Optional[IngestRun] = None
        self._write_summary = WriteSummary()

    def write(self, ingest_run: IngestRun) -> WriteSummary:
        """Capture ingest run in memory and return summary."""
        self.written_ingest_run = ingest_run

        # Track what would be written
        if ingest_run.normalized_dataset:
            sample_count = len(ingest_run.normalized_dataset.samples)
            self._write_summary.add_samples(created=sample_count)
            self._write_summary.add_datasets(created=1)
            self._write_summary.add_configs(created=1)
            self._write_summary.add_runs(created=1)

            # Count unique queries and documents
            unique_queries = set()
            unique_documents = set()
            for sample in ingest_run.normalized_dataset.samples:
                unique_queries.add(sample.judging_sample.query.query_text)
                unique_documents.add(sample.judging_sample.document.doc_text)
            self._write_summary.add_queries(created=len(unique_queries))
            self._write_summary.add_documents(created=len(unique_documents))

        return self._write_summary
