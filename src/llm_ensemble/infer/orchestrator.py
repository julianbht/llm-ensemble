"""Orchestrator for the infer CLI.

This module handles infrastructure concerns (run management, logging, manifests)
and wires up the domain service with concrete adapter implementations.
It follows hexagonal architecture by delegating business logic to the domain
service while handling all infrastructure responsibilities.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO, Callable

from llm_ensemble.infer.config_loaders import load_model_config
from llm_ensemble.infer.schemas import ModelJudgement
from llm_ensemble.infer.ports import LLMProvider, ExampleReader, JudgementWriter
from llm_ensemble.infer.domain import InferenceService
from llm_ensemble.infer.adapters.io import NdjsonExampleReader, NdjsonJudgementWriter
from llm_ensemble.infer.adapters.provider_factory import get_provider
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.logging.logger import get_logger


def run_inference(
    model: str,
    input_file: Path,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    config_dir: Optional[Path] = None,
    prompts_dir: Optional[Path] = None,
    prompt: str = "thomas-et-al-prompt",
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
    # Dependency injection (defaults to production implementations)
    example_reader: Optional[ExampleReader] = None,
    judgement_writer_factory: Optional[Callable[[Path], JudgementWriter]] = None,
    provider_factory_fn: Optional[Callable] = None,
) -> dict:
    """Run LLM inference on judging examples and output structured judgements.

    This orchestrator handles infrastructure concerns:
    - Run management (directories, IDs, manifests)
    - Logging setup and output
    - Adapter instantiation and dependency injection
    - Delegating business logic to InferenceService

    The domain service handles the pure business logic of the inference pipeline.

    Args:
        model: Model ID for .yaml config file (e.g., 'gpt-oss-20b')
        input_file: Input NDJSON file with JudgingExample records (from ingest CLI)
        run_id: Custom run ID (auto-generates if not provided)
        limit: Process at most N examples
        config_dir: Path to model configs directory
        prompts_dir: Path to prompt templates directory (defaults to configs/prompts)
        prompt: Prompt template name (without .jinja extension)
        save_logs: Save logs to run.log file in run directory
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        log_file: Optional file handle for logging (used when save_logs=True)
        example_reader: Optional ExampleReader adapter (defaults to NdjsonExampleReader)
        judgement_writer_factory: Optional factory for JudgementWriter (defaults to NdjsonJudgementWriter)
        provider_factory_fn: Optional provider factory function (defaults to get_provider)

    Returns:
        Dictionary with run metadata including:
        - run_id: The run identifier
        - run_dir: Path to run directory
        - output_file: Path to output judgements file
        - judgement_count: Total number of judgements processed
        - error_count: Number of failed judgements
        - avg_latency_ms: Average latency per judgement

    Raises:
        FileNotFoundError: If model config not found
        Exception: If inference fails
    """
    # Load model config
    model_config = load_model_config(model, config_dir)

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(model_config.model_id)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="infer", official=official)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "judgements.ndjson"

    # Set up log file if requested and not already provided
    log_file_handle = log_file
    close_log_file = False
    if save_logs and log_file_handle is None:
        log_file_path = run_dir / "run.log"
        log_file_handle = open(log_file_path, "w", encoding="utf-8")
        close_log_file = True

    # Initialize logger
    logger = get_logger("infer", run_id=run_id, log_file=log_file_handle)

    logger.info("Starting inference", model=model_config.model_id, provider=model_config.provider, prompt=prompt)
    logger.info("Run directory", path=str(run_dir))
    logger.info("Output file", path=str(output_file))

    # Instantiate adapters (dependency injection)
    reader = example_reader or NdjsonExampleReader()
    writer_factory = judgement_writer_factory or NdjsonJudgementWriter
    provider_factory = provider_factory_fn or get_provider

    writer = writer_factory(output_file)
    provider = provider_factory(model_config)

    # Create domain service with injected ports
    service = InferenceService(
        example_reader=reader,
        judgement_writer=writer,
        llm_provider=provider,
    )

    # Define logging callback for domain service
    def log_judgement(judgement: ModelJudgement) -> None:
        """Callback to log each judgement (infrastructure concern)."""
        if judgement.label is None:
            logger.warning(
                "Judgement error",
                query_id=judgement.query_id,
                docid=judgement.docid,
                warnings=judgement.warnings,
            )
        else:
            logger.info(
                "Processed judgement",
                query_id=judgement.query_id,
                docid=judgement.docid,
                label=judgement.label,
                latency_ms=f"{judgement.latency_ms:.1f}",
            )

    # Run inference via domain service
    try:
        logger.info("Reading examples", input_file=str(input_file))

        stats = service.run_inference(
            input_path=input_file,
            model_config=model_config,
            prompt_template_name=prompt,
            prompts_dir=prompts_dir,
            limit=limit,
            on_judgement=log_judgement,
        )

        logger.info(
            "Inference complete",
            total_judgements=stats["judgement_count"],
            errors=stats["error_count"],
            avg_latency_ms=f"{stats['avg_latency_ms']:.1f}",
        )

    except Exception as e:
        logger.error("Inference failed", error=str(e))
        if close_log_file and log_file_handle is not None:
            log_file_handle.close()
        raise

    # Write manifest
    write_manifest(
        run_dir=run_dir,
        cli_name="infer",
        cli_args={
            "model": model,
            "input_file": str(input_file),
            "limit": limit,
            "prompt": prompt,
        },
        metadata={
            "model_config": model_config.model_dump(),
            "prompt_template": prompt,
            "judgement_count": stats["judgement_count"],
            "error_count": stats["error_count"],
            "avg_latency_ms": stats["avg_latency_ms"],
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

    # Return combined metadata and statistics
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "output_file": output_file,
        **stats,  # Include judgement_count, error_count, avg_latency_ms
    }
