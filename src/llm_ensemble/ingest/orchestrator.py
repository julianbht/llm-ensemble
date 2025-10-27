"""Orchestrator for the ingest CLI.

This module handles infrastructure concerns for the ingestion pipeline:
- Loading configurations
- Setting up run directories and logging
- Instantiating adapters via factories
- Delegating business logic to domain service
- Writing manifests

It is separated from the CLI entry point (ingest_cli.py) for testability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.ingest.domain import IngestionService
from llm_ensemble.ingest.adapters import get_example_reader, get_example_writer
from llm_ensemble.ingest.config_loaders import load_ingest_io_config
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.logging.logger import get_logger


def run_ingest(
    io_format: str,
    data_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
) -> dict:
    """Normalize a raw IR dataset into JudgingExample NDJSON records.

    Infrastructure orchestration that coordinates:
    - Loading dataset configuration
    - Setting up run directory and logging
    - Instantiating adapters via factories
    - Delegating business logic to IngestionService
    - Writing manifests

    Args:
        dataset: Dataset config name (e.g., 'llm_judge_challenge')
        data_dir: Override data directory from config (defaults to config value)
        run_id: Custom run ID (auto-generates if not provided)
        limit: Process at most N examples
        save_logs: Save logs to run.log file in run directory
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        log_file: Optional file handle for logging (used when save_logs=True)

    Returns:
        Dictionary with run metadata including:
        - run_id: The run identifier
        - run_dir: Path to run directory
        - output_file: Path to output samples file
        - sample_count: Total number of samples processed
        - dataset_version: Version of the dataset

    Raises:
        FileNotFoundError: If dataset config not found or data directory doesn't exist
        ValueError: If adapter is not recognized or dataset files are malformed
    """
    # Load I/O config (includes dataset_id and data_dir)
    io_config = load_ingest_io_config(io_format)

    # Use data_dir override if provided, otherwise use config default
    actual_data_dir = data_dir if data_dir is not None else io_config.data_dir

    # Verify data directory exists
    if not actual_data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {actual_data_dir}")

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(io_config.dataset_id)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="ingest", official=official)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / f"samples.{io_config.io_format.replace('_ingest', '')}"

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

    # Instantiate adapters via factories
    example_reader = get_example_reader(io_config)
    example_writer = get_example_writer(io_config, output_file)

    # Create domain service
    service = IngestionService(
        example_reader=example_reader,
        example_writer=example_writer,
    )

    # Run ingestion pipeline (pure business logic)
    try:
        stats = service.run_ingestion(
            data_dir=actual_data_dir,
            limit=limit,
        )
        count = stats["sample_count"]
    except Exception as e:
        logger.error("Ingest failed", error=str(e))
        if close_log_file and log_file_handle is not None:
            log_file_handle.close()
        raise

    logger.info("Ingest complete", total_examples=count)
    
    # Write manifest
    write_manifest(
        run_dir=run_dir,
        cli_name="ingest",
        cli_args={
            "dataset_id": io_config.dataset_id,
            "io_format": io_config.io_format,
            "data_dir": str(actual_data_dir),
            "limit": limit,
        },
        metadata={
            "sample_count": count,
            "output_file": str(output_file),
        },
        official=official,
        notes=notes,
    )
    
    logger.info("Manifest written", path=str(run_dir / "manifest.json"))
    
    # Close log file if we opened it
    if close_log_file and log_file_handle is not None:
        logger.info("Logs saved", path=str(run_dir / "run.log"))
        log_file_handle.close()
    
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "output_file": output_file,
        "sample_count": count,
        "dataset_id": io_config.dataset_id,
    }
