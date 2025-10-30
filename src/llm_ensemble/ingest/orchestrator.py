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
from typing import Optional, TextIO

from llm_ensemble.ingest.domain import IngestionService
from llm_ensemble.ingest.adapters import get_sample_reader, get_dataset_writer
from llm_ensemble.ingest.config_loaders import load_ingest_io_config
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.run_id import generate_run_id
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.runtime.git_utils import get_git_info
from llm_ensemble.libs.logging.logger import get_logger


def run_ingest(
    input_path: Path,
    io_format: str,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
) -> None:
    """Normalize a raw IR dataset into judging samples with full provenance.

    Infrastructure orchestration that coordinates:
    - Loading I/O configuration
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters via factories
    - Reading samples, attaching manifest to each sample, and writing output

    Args:
        input_path: Path to input directory containing raw dataset files
        io_format: I/O format config name (e.g., 'llm_judge_challenge')
        run_id: Custom run ID (auto-generates if not provided)
        limit: Process at most N samples
        save_logs: Save logs to run.log file in run directory
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        log_file: Optional file handle for logging (used when save_logs=True)

    Raises:
        FileNotFoundError: If I/O config not found or input path doesn't exist
        ValueError: If adapter is not recognized or dataset files are malformed
    """
    # Load I/O config (includes dataset_id and adapter specifications)
    io_config = load_ingest_io_config(io_format)

    # Verify input directory exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Generate or use provided run_id
    actual_run_id = run_id or generate_run_id(io_config.dataset_id)

    # Get run directory path and create it
    run_dir = PathManager.get_run_dir(
        cli_name="ingest",
        run_id=actual_run_id,
        official=official
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get git info for reproducibility
    git_info = get_git_info()

    # Create immutable run info (runtime context known before run starts)
    run_info = IngestRunInfo(
        run_id=actual_run_id,
        run_type="official" if official else "test",
        notes=notes,
        git_sha=git_info["git_sha"],
        git_clean=git_info["git_clean"],
        git_branch=git_info["git_branch"],
        io_config_name=io_format,
        io_config=io_config,
        input_path=str(input_path),
        limit=limit,
    )

    # Set up log file if requested and not already provided
    log_file_handle = log_file
    close_log_file = False
    if save_logs and log_file_handle is None:
        log_file_path = run_dir / "run.log"
        log_file_handle = open(log_file_path, "w", encoding="utf-8")
        close_log_file = True

    # Initialize logger
    logger = get_logger("ingest", run_id=actual_run_id, log_file=log_file_handle)

    logger.info(
        "Starting ingest",
        dataset_id=io_config.dataset_id,
        io_format=io_config.io_format,
        input_path=str(input_path),
        limit=limit,
    )
    logger.info("Run directory", path=str(run_dir))

    # Instantiate adapters via factories
    sample_reader = get_sample_reader(io_config)
    dataset_writer = get_dataset_writer(io_config)

    # Create domain service
    service = IngestionService(
        sample_reader=sample_reader,
        dataset_writer=dataset_writer,
    )

    # Run ingestion pipeline (pure business logic)
    try:
        summary = service.ingest_dataset(
            data_dir=input_path,
            run_info=run_info,
            run_dir=run_dir,
            limit=limit,
        )
        sample_count = summary.sample_count
        logger.info("Samples processed", count=sample_count)

        # Write standalone summary.json for convenience (not source of truth)
        write_standalone_summary(summary, run_dir)
        logger.info("Summary written", path=str(run_dir / "summary.json"))

    except Exception as e:
        logger.error("Ingest failed", error=str(e))
        if close_log_file and log_file_handle is not None:
            log_file_handle.close()
        raise

    logger.info("Ingest complete", total_samples=sample_count)

    # Close log file if we opened it
    if close_log_file and log_file_handle is not None:
        logger.info("Logs saved", path=str(run_dir / "run.log"))
        log_file_handle.close()