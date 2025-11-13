"""Orchestrator for the ingest CLI.

This module handles infrastructure concerns for the ingestion pipeline:
- Loading configurations
- Setting up run directories and logging
- Building manifests with git info and execution parameters
- Instantiating adapters via factories
- Delegating business logic to domain service (which sets timing and finalizes manifest)
It is separated from the CLI entry point (ingest_cli.py) for testability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.domain import IngestionService
from llm_ensemble.ingest.schemas import WriteSummary
from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.libs.schemas import LoggingConfig
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.libs.logging.log_events import IngestLogEvent
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.runtime.git_utils import get_git_info
from llm_ensemble.libs.logging import configure_logger


def run_ingest(
    io_config: IOConfig,
    logging_config: LoggingConfig,
    io_config_name: str,
    input_path: Path,
    run_name: Optional[str] = None,
    limit: Optional[int] = None,
    official: bool = False,
    notes: Optional[str] = None,
) -> None:
    """Normalize a raw IR dataset into judging samples with full provenance.

    Infrastructure orchestration that coordinates:
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters via factories
    - Reading samples, attaching manifest to each sample, and writing output

    Config is provided as a final, validated object (CLI handles loading and overrides).

    Args:
        io_config: Ingest-specific I/O configuration (already loaded and validated with overrides applied)
        logging_config: Logging configuration (controls pretty printing and log saving)
        io_config_name: Name of the I/O config file (e.g., "llm_judge_challenge_ndjson")
        input_path: Path to input directory containing raw dataset files
        run_name: Custom run ID (auto-generates if not provided)
        limit: Process at most N samples
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)

    Raises:
        FileNotFoundError: If input path doesn't exist
        ValueError: If adapter is not recognized or dataset files are malformed
    """

    # Verify input directory exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Generate or use provided run_name (collect name hints from config)
    if run_name is None:
        run_name = generate_run_name([io_config.name_hint])

    # Get git info for reproducibility
    git_info = get_git_info()

    # Create immutable run info using create() method (runtime context known before run starts)
    run_info = IngestRunInfo.create(
        run_name=run_name,
        io_config_name=io_config_name,
        io_config=io_config,
        input_path=str(input_path),
        limit=limit,
        run_type=RunType.OFFICIAL if official else RunType.TEST,
        notes=notes,
        git_sha=git_info["git_sha"],
        git_clean=git_info["git_clean"],
        git_branch=git_info["git_branch"],
    )

    # Get run directory from run_info and create it
    run_dir = run_info.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up log file path if saving logs
    log_file_path = run_dir / "run.log" if logging_config.save_logs else None

    # Initialize structlog logger with config
    logger = configure_logger(
        cli_name="ingest",
        run_name=run_name,
        pretty_print=logging_config.pretty_print,
        save_logs=logging_config.save_logs,
        log_file_path=log_file_path,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
    )

    logger.info(
        IngestLogEvent.INGEST_STARTED,
        io_format=io_config_name,
        input_path=str(input_path),
        limit=limit,
    )
    logger.info(IngestLogEvent.RUN_DIRECTORY_CREATED, path=str(run_dir))

    # Instantiate adapters directly from config
    dataset_reader = io_config.get_reader()
    dataset_writer = io_config.get_writer()

    # Create domain service
    service = IngestionService(
        dataset_reader=dataset_reader,
        dataset_writer=dataset_writer,
    )

    # Define logging callback for write operations (infrastructure concern)
    def on_write(write_summary: WriteSummary) -> None:
        """Log write results using WriteSummary from writer."""
        # Use the WriteSummary returned by the writer for consistent logging
        for log_entry in write_summary.get_log_entries():
            logger.info(**log_entry)

    # Run ingestion pipeline (pure business logic)
    try:
        summary = service.ingest_dataset(
            data_dir=input_path,
            run_info=run_info,
            limit=limit,
            on_write=on_write,
        )

        # Write standalone summary.json for convenience (not source of truth)
        write_standalone_summary(summary, run_dir)
        logger.info(IngestLogEvent.INGEST_SUMMARY_WRITTEN, path=str(run_dir / "summary.json"))

    except Exception as e:
        logger.error(IngestLogEvent.INGEST_FAILED, error=str(e))
        raise

    logger.info(IngestLogEvent.INGEST_COMPLETE)

    # Log where logs were saved if enabled
    if logging_config.save_logs:
        logger.info(IngestLogEvent.LOGS_SAVED, path=str(run_dir / "run.log"))