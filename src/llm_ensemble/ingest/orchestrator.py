"""Orchestrator for the ingest CLI.

This module contains the top-level orchestration logic for ingesting and normalizing
raw IR datasets into JudgingExample records. It is separated from the CLI entry point
(ingest_cli.py) to enable better testability and reusability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.ingest.adapters import load_adapter
from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.libs.config import load_dataset_config
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.logging.logger import get_logger


def _json_dumps(obj: JudgingExample) -> str:
    """Serialize JudgingExample to JSON."""
    return obj.model_dump_json()


def run_ingest(
    dataset: str,
    data_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
) -> dict:
    """Normalize a raw IR dataset into JudgingExample NDJSON records.
    
    This is the main orchestration function that coordinates:
    - Loading dataset configuration
    - Loading the appropriate adapter
    - Setting up run directory and output files
    - Processing examples through the adapter
    - Writing results and manifest
    
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
        ImportError: If adapter cannot be loaded
        AttributeError: If adapter is malformed
    """
    # Load dataset config
    config = load_dataset_config(dataset)
    
    # Use data_dir override if provided, otherwise use config default
    actual_data_dir = data_dir if data_dir is not None else config.data_dir
    
    # Verify data directory exists
    if not actual_data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {actual_data_dir}")
    
    # Load the adapter dynamically
    iter_examples = load_adapter(config.adapter)
    
    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(config.dataset_id)
    
    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="ingest", official=official)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "samples.ndjson"
    
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
        dataset_id=config.dataset_id,
        adapter=config.adapter,
        data_dir=str(actual_data_dir),
        limit=limit,
    )
    logger.info("Run directory", path=str(run_dir))
    logger.info("Output file", path=str(output_file))
    
    # Process examples
    count = 0
    try:
        with output_file.open("w", encoding="utf-8", newline="\n") as sink:
            for ex in iter_examples(actual_data_dir):
                sink.write(_json_dumps(ex) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break
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
            "dataset_id": config.dataset_id,
            "adapter": config.adapter,
            "data_dir": str(actual_data_dir),
            "limit": limit,
        },
        metadata={
            "sample_count": count,
            "output_file": str(output_file),
            "dataset_version": config.version,
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
        "dataset_version": config.version,
    }
