"""Orchestrator for the infer CLI.

This module handles infrastructure concerns for the inference pipeline:
- Loading configurations
- Setting up run directories and logging
- Building manifests with git info and execution parameters
- Instantiating adapters dynamically from config specifications
- Delegating business logic to domain service (which sets timing and finalizes manifest)

It is separated from the CLI entry point (infer_cli.py) for testability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.infer.domain import InferenceService
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.run_id import generate_run_id
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.runtime.git_utils import get_git_info
from llm_ensemble.libs.logging.logger import get_logger


def run_inference(
    model_config: ModelConfig,
    prompt_config: PromptConfig,
    io_config: IOConfig,
    input_file: Path,
    model_config_name: str,
    prompt_config_name: str,
    io_config_name: str,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
) -> None:
    """Run LLM inference on judging examples with full provenance.

    Infrastructure orchestration that coordinates:
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters dynamically from config specifications
    - Running inference, attaching manifest metadata to each judgement, and writing output

    Configs are provided as final, validated objects (CLI handles loading and overrides).

    Args:
        model_config: Model configuration (already loaded and validated with overrides applied)
        prompt_config: Prompt configuration (already loaded and validated with overrides applied)
        io_config: I/O configuration (already loaded and validated with overrides applied)
        input_file: Input file with JudgingExample records (from ingest CLI)
        model_config_name: Name of the model config file (e.g., "gpt-oss-20b")
        prompt_config_name: Name of the prompt config file (e.g., "thomas-et-al-prompt")
        io_config_name: Name of the I/O config file (e.g., "ndjson")
        run_id: Custom run ID (auto-generates if not provided)
        limit: Process at most N examples
        save_logs: Save logs to run.log file in run directory
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        log_file: Optional file handle for logging (used when save_logs=True)

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If adapter is not recognized or config is invalid
    """

    # Verify input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Generate or use provided run_id (collect name hints from all configs)
    if run_id is None:
        run_id = generate_run_id([
            model_config.name_hint,
            prompt_config.name_hint,
            io_config.name_hint,
        ])

    # Get run directory path and create it
    run_dir = PathManager.get_run_dir(
        cli_name="infer",
        run_id=run_id,
        official=official
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get git info for reproducibility
    git_info = get_git_info()

    # Create immutable run info (runtime context known before run starts)
    run_info = InferRunInfo(
        run_id=run_id,
        run_type="official" if official else "test",
        notes=notes,
        git_sha=git_info["git_sha"],
        git_clean=git_info["git_clean"],
        git_branch=git_info["git_branch"],
        model_config_name=model_config_name,
        prompt_config_name=prompt_config_name,
        io_config_name=io_config_name,
        model_cfg=model_config,
        prompt_config=prompt_config,
        io_config=io_config,
        input_file=str(input_file),
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
    logger = get_logger("infer", run_id=run_id, log_file=log_file_handle)

    logger.info(
        "Starting inference",
        model=model_config_name,
        provider=model_config.provider,
        io_format=io_config_name,
        prompt=prompt_config_name,
        input_file=str(input_file),
        limit=limit,
    )
    logger.info("Run directory", path=str(run_dir))

    # Instantiate I/O adapters directly from config
    reader = io_config.get_reader()
    writer = io_config.get_writer()

    # Instantiate prompt builder and response parser directly from config
    # Builder is responsible for loading its own template
    prompt_builder = prompt_config.get_prompt_builder()
    response_parser = prompt_config.get_response_parser(score_field="O")

    # Instantiate provider directly from config
    provider = model_config.get_provider()

    # Create domain service - it orchestrates ALL port interactions
    service = InferenceService(
        example_reader=reader,
        judgement_writer=writer,
        prompt_builder=prompt_builder,
        llm_provider=provider,
        response_parser=response_parser,
        logger=logger,
    )

    # Run inference pipeline (pure business logic)
    try:
        summary = service.run_inference(
            input_path=input_file,
            model_config=model_config,
            run_info=run_info,
            run_dir=run_dir,
            limit=limit,
        )
        judgement_count = summary.judgement_count
        logger.info("Judgements processed", count=judgement_count)

        # Write standalone summary.json for convenience (not source of truth)
        write_standalone_summary(summary, run_dir)
        logger.info("Summary written", path=str(run_dir / "summary.json"))

    except Exception as e:
        logger.error("Inference failed", error=str(e))
        if close_log_file and log_file_handle is not None:
            log_file_handle.close()
        raise

    logger.info(
        "Inference complete",
        total_judgements=summary.judgement_count,
        parsing_failures=summary.error_count,
        avg_latency_ms=f"{summary.avg_latency_ms:.1f}",
    )

    # Log warnings summary if any warnings were collected
    if summary.warnings_summary and sum(summary.warnings_summary.values()) > 0:
        total_warnings = sum(summary.warnings_summary.values())
        logger.info(
            f"⚠ Warnings collected: {total_warnings} total",
            **summary.warnings_summary
        )

    # Close log file if we opened it
    if close_log_file and log_file_handle is not None:
        logger.info("Logs saved", path=str(run_dir / "run.log"))
        log_file_handle.close()
