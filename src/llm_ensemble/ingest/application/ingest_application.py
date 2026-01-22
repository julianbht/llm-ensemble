from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.domain.ingest_run_factory import IngestRunFactory

# Driving port (application implements this)
from llm_ensemble.ingest.application.ports.driving.for_running_ingest import (
    ForRunningIngest,
)

# Driven ports (application depends on these)
from llm_ensemble.ingest.application.ports.driven.for_input import ForInput
from llm_ensemble.ingest.application.ports.driven.for_output import ForOutput

from llm_ensemble.libs.logging.structlog_logger import get_logger
from llm_ensemble.libs.logging.log_events import IngestLogEvent
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.runtime.run_manager import persist_run_summary


class IngestApplication(ForRunningIngest):
    """
    Application use case for coordinating data ingestion pipeline.
    Implements the driving port ForRunningIngest - this IS the application's API.
    Driving adapters (CLI, Web API, etc.) call the run_ingest() method.
    """

    def __init__(
        self,
        reader: ForInput,
        writer: ForOutput,
        run_dir: Path,
        run_name: str,
        io_name: str,
    ):
        """Initialize ingest use case with port dependencies.

        Args:
            dataset_reader: Port for reading and normalizing datasets
            dataset_writer: Port for writing judging samples
            run_dir: Run directory path
            run_name: Run identifier
            io_name: Name of the I/O configuration used
        """
        self.input_port = reader
        self.output_port = writer
        self.run_dir = run_dir
        self.run_name = run_name
        self.io_name = io_name

    def run_ingest(
        self,
        input_path: Path,
        limit: Optional[int],
        official: bool,
        notes: Optional[str],
    ) -> IngestRunSummary:
        """
        Execute the complete ingestion pipeline.

        Args:
            input_path: Path to input directory containing raw dataset files
            limit: Process at most N samples
            official: Mark as official run
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            IngestRunSummary with statistics and timing

        Raises:
            FileNotFoundError: If input path doesn't exist
            ValueError: If adapter is not recognized or dataset files are malformed
            Exception: If any step in the pipeline fails
        """
        # Get logger
        logger = get_logger()
        logger.info(IngestLogEvent.INGEST_STARTED, name=self.run_name)

        start_time = datetime.now()

        # Read in dataset
        normalized_dataset: NormalizedDataset = self.input_port.read(
            input_path, limit=limit
        )
        logger.info(IngestLogEvent.READ_COMPLETE)

        end_time = datetime.now()

        # Build output
        ingest_run = IngestRunFactory.create(
            io_config_name=self.io_name,
            input_path=input_path,
            limit=limit,
            run_name=self.run_name,
            run_type=RunType.OFFICIAL if official else RunType.TEST,
            normalized_dataset=normalized_dataset,
            start_time=start_time,
            end_time=end_time,
            notes=notes,
        )

        # Write output
        write_summary = self.output_port.write(ingest_run)
        logger.info(IngestLogEvent.PERSISTENCE_COMPLETE)

        # Build run summary (clear dataset to avoid duplication)
        ingest_run.normalized_dataset = None  # Dataset stats already in summary metrics
        run_summary = IngestRunSummary(
            start_time=start_time,
            end_time=end_time,
            run=ingest_run,
            sample_count=normalized_dataset.sample_count,
            write_summary=write_summary,
        )

        # Write summary.json
        run_summary_path = persist_run_summary(run_summary, self.run_dir)
        logger.info(
            IngestLogEvent.INGEST_RUN_SUMMARY_WRITTEN, path=str(run_summary_path)
        )

        # Return summary
        return run_summary
