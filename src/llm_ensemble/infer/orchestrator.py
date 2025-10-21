"""Orchestrator for the infer CLI.

This module handles infrastructure concerns (run management, logging, manifests)
and wires up the domain service with concrete adapter implementations.
It follows hexagonal architecture by delegating business logic to the domain
service while handling all infrastructure responsibilities.

All adapters are instantiated via explicit configuration - no implicit defaults.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.infer.config_loaders import load_model_config, load_io_config, load_prompt_config
from llm_ensemble.infer.schemas import ModelJudgement
from llm_ensemble.infer.domain import InferenceService
from llm_ensemble.infer.adapters.io_factory import get_example_reader, get_judgement_writer
from llm_ensemble.infer.adapters.provider_factory import get_provider
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.logging.logger import get_logger
from llm_ensemble.libs.utils.config_overrides import apply_overrides


def run_inference(
    model: str,
    input_file: Path,
    io_format: str = "ndjson",
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    config_dir: Optional[Path] = None,
    io_dir: Optional[Path] = None,
    prompts_dir: Optional[Path] = None,
    prompt: str = "thomas-et-al-prompt",
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
    config_overrides: Optional[dict] = None,
) -> dict:
    """Run LLM inference on judging examples and output structured judgements.

    This orchestrator handles infrastructure concerns:
    - Run management (directories, IDs, manifests)
    - Logging setup and output
    - Adapter instantiation via explicit configuration
    - Delegating business logic to InferenceService

    The domain service handles the pure business logic of the inference pipeline.

    All behavior is explicitly configured - no implicit defaults.

    Args:
        model: Model ID for .yaml config file (e.g., 'gpt-oss-20b' for configs/models/gpt-oss-20b.yaml)
        input_file: Input file with JudgingExample records (from ingest CLI)
        io_format: I/O format config name (e.g., 'ndjson' for configs/io/ndjson.yaml)
        run_id: Custom run ID (auto-generates if not provided)
        limit: Process at most N examples
        config_dir: Path to model configs directory (defaults to configs/models)
        io_dir: Path to I/O configs directory (defaults to configs/io)
        prompts_dir: Path to prompt templates directory (defaults to configs/prompts)
        prompt: Prompt template name (without .jinja extension)
        save_logs: Save logs to run.log file in run directory
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        log_file: Optional file handle for logging (used when save_logs=True)
        config_overrides: Optional dict of config overrides (e.g., {"default_params": {"temperature": 0.7}})

    Returns:
        Dictionary with run metadata including:
        - run_id: The run identifier
        - run_dir: Path to run directory
        - output_file: Path to output judgements file
        - judgement_count: Total number of judgements processed
        - error_count: Number of failed judgements
        - avg_latency_ms: Average latency per judgement

    Raises:
        FileNotFoundError: If model/io config not found
        ValueError: If config validation fails
        Exception: If inference fails
    """
    # Load configurations
    model_config = load_model_config(model, config_dir)
    io_config = load_io_config(io_format, io_dir)
    prompt_config = load_prompt_config(prompt, prompts_dir)

    # Apply overrides if provided by cli
    if config_overrides:
        # Separate overrides by config type based on keys
        model_overrides = {}
        io_overrides = {}
        prompt_overrides = {}

        for key, value in config_overrides.items():
            # Model config fields: provider, default_params, context_window, etc.
            if key in ["provider", "context_window", "default_params", "capabilities",
                       "hf_endpoint_url", "hf_model_name", "openrouter_model_id"]:
                model_overrides[key] = value
            # I/O config fields: reader, writer
            elif key in ["reader", "writer"]:
                io_overrides[key] = value
            # Prompt config fields: variables, prompt_builder, response_parser, etc.
            elif key in ["variables", "prompt_builder", "response_parser", "template_file"]:
                prompt_overrides[key] = value
            else:
                # Try model first (most common), will fail with validation error if wrong
                model_overrides[key] = value

        # Apply overrides to each config
        if model_overrides:
            model_config = apply_overrides(model_config, model_overrides)
        if io_overrides:
            io_config = apply_overrides(io_config, io_overrides)
        if prompt_overrides:
            prompt_config = apply_overrides(prompt_config, prompt_overrides)

    # Create or use provided run ID
    if run_id is None:
        run_id = create_run_id(model_config.model_id)

    # Set up run directory and output file
    run_dir = get_run_dir(run_id, cli_name="infer", official=official)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / f"judgements.{io_config.io_format}"

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
        model=model_config.model_id,
        provider=model_config.provider,
        io_format=io_config.io_format,
        prompt=prompt,
        overrides=config_overrides if config_overrides else None,
    )
    logger.info("Run directory", path=str(run_dir))
    logger.info("Output file", path=str(output_file))

    # Instantiate adapters via explicit configuration (no defaults)
    reader = get_example_reader(io_config)
    writer = get_judgement_writer(io_config, output_file)
    provider = get_provider(model_config)

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
    manifest_metadata = {
        "model_config": model_config.model_dump(),
        "io_config": io_config.model_dump(),
        "prompt_config": prompt_config.model_dump(),
        "judgement_count": stats["judgement_count"],
        "error_count": stats["error_count"],
        "avg_latency_ms": stats["avg_latency_ms"],
        "output_file": str(output_file),
    }

    # Track overrides for full reproducibility
    if config_overrides:
        manifest_metadata["config_overrides"] = config_overrides

    write_manifest(
        run_dir=run_dir,
        cli_name="infer",
        cli_args={
            "model": model,
            "input_file": str(input_file),
            "io_format": io_format,
            "limit": limit,
            "prompt": prompt,
        },
        metadata=manifest_metadata,
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
