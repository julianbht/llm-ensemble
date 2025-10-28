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
from llm_ensemble.libs.runtime.manifest_manager import (
    initialize_manifest,
    write_standalone_manifest,
)
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
    """Normalize a raw IR dataset into judging samples with full provenance.

    Infrastructure orchestration that coordinates:
    - Loading I/O configuration
    - Applying config overrides
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters via factories
    - Reading samples, attaching manifest to each sample, and writing output

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

    # Initialize manifest builder (creates run_id and directory)
    manifest_builder = initialize_manifest(
        cli_name="ingest",
        name_hint=io_config.dataset_id,
        run_id=run_id,  # None = auto-generate, otherwise use provided
        official=official,
        notes=notes,
    )

    # Extract run info for logging and paths
    run_id = manifest_builder.run_id
    run_dir = manifest_builder.run_dir

    # Add ingest-specific fields to manifest builder
    manifest_builder.add("io_config_name", io_format)
    manifest_builder.add("io_config", io_config)
    manifest_builder.add("limit", limit)
    manifest_builder.add("config_overrides", config_overrides or {})

    # Set up output file
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
        manifest = service.ingest_dataset(
            data_dir=actual_data_dir,
            manifest_builder=manifest_builder,
            output_path=output_file,
            limit=limit,
        )
        sample_count = manifest.sample_count
        logger.info("Samples processed", count=sample_count)

        # Write standalone manifest.json for convenience (not source of truth)
        write_standalone_manifest(manifest, run_dir)
        logger.info("Manifest written", path=str(run_dir / "manifest.json"))

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
