"""Factory for creating IngestRun entities.

Domain Layer - Factory Pattern

Creates IngestRun aggregate root from domain entities and primitive values,
assembling the complete record for manifest persistence.

This factory belongs in the domain layer because it only depends on
domain entities and performs pure assembly logic. The application layer
is responsible for providing these values.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.domain.entities.ingest_run import IngestRun
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.libs.runtime.run_info import RunType


class IngestRunFactory:
    """Factory for creating IngestRun aggregate root from domain entities.

    Domain layer factory - pure assembly logic with no adapter dependencies."""

    @staticmethod
    def create(
        io_config_name: str,
        input_path: Path,
        limit: Optional[int],
        run_name: str,
        run_type: RunType,
        normalized_dataset: NormalizedDataset,
        start_time: datetime,
        end_time: datetime,
        notes: Optional[str],
    ) -> IngestRun:
        """Create IngestRun aggregate root from domain entities and primitive values.

        Args:
            io_config_name: Name of the I/O configuration used
            input_path: Path to input directory containing raw dataset files
            limit: Maximum number of samples to process (None = no limit)
            run_name: Run identifier
            run_type: Type of run (OFFICIAL or TEST)
            normalized_dataset: Dataset produced by this run
            start_time: When the run started
            end_time: When the run completed
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            Assembled IngestRun aggregate root
        """
        # Build IngestRunConfig entity
        run_config = IngestRunConfig(
            io_config_name=io_config_name,
            input_path=str(input_path),
            limit=limit,
        )

        # Assemble complete aggregate root
        return IngestRun(
            run_name=run_name,
            run_type=run_type,
            ingest_run_config=run_config,
            normalized_dataset=normalized_dataset,
            start_time=start_time,
            end_time=end_time,
            notes=notes,
        )
