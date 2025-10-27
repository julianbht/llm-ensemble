"""Orchestrator for the ingest CLI.

This module handles infrastructure concerns for the ingestion pipeline:
- Loading configurations
- Setting up run directories and logging
- Building manifests with git info and execution parameters
- Instantiating adapters via factories
- Delegating business logic to domain service

It is separated from the CLI entry point (ingest_cli.py) for testability.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
from uuid import uuid4

from llm_ensemble.ingest.schemas import IngestManifest
from llm_ensemble.ingest.domain import IngestionService
from llm_ensemble.ingest.adapters import get_sample_reader, get_dataset_writer
from llm_ensemble.ingest.config_loaders import load_ingest_io_config
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir
from llm_ensemble.libs.runtime.git_utils import get_git_info
from llm_ensemble.libs.logging.logger import get_logger
from llm_ensemble.libs.utils.config_overrides import apply_overrides


def run_ingest(
    io_format: str,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
    config_overrides: Optional[dict] = None,
) -> dict:
    """Normalize a raw IR dataset into a NormalizedDataset with full provenance.

    Infrastructure orchestration that coordinates:
    - Loading I/O configuration
    - Applying config overrides
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters via factories
    - Reading samples, building NormalizedDataset, and writing output

    Args:
        io_format: I/O format config name (e.g., 'llm_judge_ingest')
        run_id: Custom run ID (auto-generates if not provided)
        limit: Process at most N samples
        save_logs: Save logs to run.log file in run directory
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        log_file: Optional file handle for logging (used when save_logs=True)
        config_overrides: Optional dict of config overrides (e.g., {"data_dir": "/custom/path"})

    Returns:
        Dictionary with run metadata including:
        - run_id: The run identifier
        - run_dir: Path to run directory
        - output_file: Path to output dataset file
        - sample_count: Total number of samples processed
        - dataset_id: Dataset identifier

    Raises:
        FileNotFoundError: If I/O config not found or data directory doesn't exist
        ValueError: If adapter is not recognized or dataset files are malformed
    """
    # Capture start time
    start_time = datetime.now()

    # Load I/O config (includes dataset_id and data_dir)
    io_config = load_ingest_io_config(io_format)

    # Apply overrides if provided
    if config_overrides:
        io_config = apply_overrides(io_config, config_overrides)

    # Use data directory from config
    actual_data_dir = io_config.data_dir

    # Verify data directory exists
    if not actual_data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {actual_data_dir}")

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(io_config.dataset_id)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="ingest", official=official)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "normalized_dataset.ndjson"

    # Set up log file if requested and not already provided
    log_file_handle = log_file
    close_log_file = False
    if save_logs and log_file_handle is None:
        log_file_path = run_dir / "run.log"
        log_file_handle = open(log_file_path, "w", encoding="utf-8")
        close_log_file = True

    # Initialize logger
    logger = get_logger("ingest", run_id=run_id, log_file=log_file_handle)

    logger.info(
        "Starting ingest",
        dataset_id=io_config.dataset_id,
        io_format=io_config.io_format,
        data_dir=str(actual_data_dir),
        limit=limit,
    )
    logger.info("Run directory", path=str(run_dir))
    logger.info("Output file", path=str(output_file))

    # Capture git info
    git_info = get_git_info()

    # Build IngestManifest (sample_count will be filled by service)
    manifest = IngestManifest(
        run_id=uuid4(),
        run_type="official" if official else "test",
        cli_name="ingest",
        start_time=start_time,
        end_time=None,  # Will be set after ingestion completes
        notes=notes,
        git_sha=git_info["git_sha"],
        git_clean=git_info["git_clean"],
        git_branch=git_info["git_branch"],
        io_config_name=io_format,
        io_config=io_config,
        limit=limit,
        config_overrides=config_overrides or {},
        sample_count=None,  # Will be set by service after reading samples
    )

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
        stats = service.ingest_dataset(
            data_dir=actual_data_dir,
            manifest=manifest,
            output_path=output_file,
            limit=limit,
        )
        sample_count = stats["sample_count"]
        logger.info("Samples processed", count=sample_count)

        # Update manifest with end time
        manifest.end_time = datetime.now()

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

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "output_file": output_file,
        "sample_count": sample_count,
        "dataset_id": io_config.dataset_id,
    }
