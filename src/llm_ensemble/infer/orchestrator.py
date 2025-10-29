"""Orchestrator for the infer CLI.

This module handles infrastructure concerns for the inference pipeline:
- Loading configurations
- Setting up run directories and logging
- Building manifests with git info and execution parameters
- Instantiating adapters via factories
- Delegating business logic to domain service (which sets timing and finalizes manifest)

It is separated from the CLI entry point (infer_cli.py) for testability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.infer.config_loaders import load_model_config, load_io_config, load_prompt_config
from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain import InferenceService
from llm_ensemble.infer.adapters.io_factory import get_example_reader, get_judgement_writer
from llm_ensemble.infer.adapters.provider_factory import get_provider
from llm_ensemble.infer.adapters.prompt_builder_factory import get_prompt_builder
from llm_ensemble.infer.adapters.response_parser_factory import get_response_parser
from llm_ensemble.libs.runtime.manifest_manager import (
    ManifestBuilder,
    write_standalone_manifest,
)
from llm_ensemble.libs.runtime.run_id import generate_run_id
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.logging.logger import get_logger
from llm_ensemble.libs.utils.config_overrides import apply_overrides


def run_inference(
    model: str,
    input_file: Path,
    prompt: str,
    io_format: str = "ndjson",
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    save_logs: bool = False,
    official: bool = False,
    notes: Optional[str] = None,
    log_file: Optional[TextIO] = None,
    config_overrides: Optional[dict] = None,
) -> None:
    """Run LLM inference on judging examples with full provenance.

    Infrastructure orchestration that coordinates:
    - Loading configurations
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters via factories
    - Running inference, attaching manifest metadata to each judgement, and writing output

    Args:
        model: Model ID for .yaml config file (e.g., 'gpt-oss-20b')
        input_file: Input file with JudgingExample records (from ingest CLI)
        prompt: Prompt template name (without .jinja extension)
        io_format: I/O format config name (e.g., 'ndjson')
        run_id: Custom run ID (auto-generates if not provided)
        limit: Process at most N examples
        save_logs: Save logs to run.log file in run directory
        official: Mark as official run (saved to official/ subdirectory for git tracking)
        notes: Notes about this run (experiment purpose, hypothesis, etc.)
        log_file: Optional file handle for logging (used when save_logs=True)
        config_overrides: Optional dict of config overrides

    Raises:
        FileNotFoundError: If config not found or input file doesn't exist
        ValueError: If adapter is not recognized or config is invalid
    """
    # Load configurations (config loaders handle directory locations)
    model_config = load_model_config(model)
    io_config = load_io_config(io_format)
    prompt_config = load_prompt_config(prompt)

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

    # Verify input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Generate or use provided run_id
    actual_run_id = run_id or generate_run_id(model_config.model_id)

    # Get run directory path and create it
    run_dir = PathManager.get_run_dir(
        cli_name="infer",
        run_id=actual_run_id,
        official=official
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize manifest builder (pure manifest construction)
    manifest_builder = ManifestBuilder(
        run_id=actual_run_id,
        run_dir=run_dir,
        cli_name="infer",
        official=official,
        notes=notes,
    )

    # Add infer-specific fields to manifest builder
    manifest_builder.add("model_config_name", model)
    manifest_builder.add("prompt_config_name", prompt)
    manifest_builder.add("io_config_name", io_format)
    manifest_builder.add("model_cfg", model_config)
    manifest_builder.add("prompt_config", prompt_config)
    manifest_builder.add("io_config", io_config)
    manifest_builder.add("input_file", str(input_file))
    manifest_builder.add("limit", limit)

    # Set up log file if requested and not already provided
    log_file_handle = log_file
    close_log_file = False
    if save_logs and log_file_handle is None:
        log_file_path = run_dir / "run.log"
        log_file_handle = open(log_file_path, "w", encoding="utf-8")
        close_log_file = True

    # Initialize logger
    logger = get_logger("infer", run_id=actual_run_id, log_file=log_file_handle)

    logger.info(
        "Starting inference",
        model=model_config.model_id,
        provider=model_config.provider,
        io_format=io_config.io_format,
        prompt=prompt,
        input_file=str(input_file),
        limit=limit,
    )
    logger.info("Run directory", path=str(run_dir))

    # Instantiate adapters via factories
    reader = get_example_reader(io_config)
    writer = get_judgement_writer(io_config)

    # Instantiate prompt builder and response parser from prompt config
    prompt_builder = get_prompt_builder(prompt_config)
    response_parser = get_response_parser(prompt_config)

    # Instantiate provider with injected prompt builder (provider only handles raw responses)
    provider = get_provider(model_config, prompt_builder)

    # Create domain service with response parser
    service = InferenceService(
        example_reader=reader,
        judgement_writer=writer,
        llm_provider=provider,
        response_parser=response_parser,
    )

    # Define logging callback for domain service
    def log_judgement(judgement: LLMJudgement) -> None:
        """Callback to log each judgement (infrastructure concern)."""
        if judgement.llm_score is None or judgement.llm_score.label is None:
            logger.warning(
                "Judgement error",
                query_text=judgement.sample.query.query_text,
                doc_text=judgement.sample.document.text,
                warnings=judgement.llm_response.warnings,
            )
        else:
            logger.info(
                "Processed judgement",
                query_text=judgement.sample.query.query_text,
                doc_text=judgement.sample.document.text,
                llm_score=judgement.llm_score.label.value,
                gold_score=judgement.sample.gold_score.value,
                latency_ms=f"{judgement.llm_response.latency_ms:.1f}",
            )

    # Run inference pipeline (pure business logic)
    try:
        manifest = service.run_inference(
            input_path=input_file,
            model_config=model_config,
            manifest_builder=manifest_builder,
            run_dir=run_dir,
            limit=limit,
            on_judgement=log_judgement,
        )
        judgement_count = manifest.judgement_count
        logger.info("Judgements processed", count=judgement_count)

        # Write standalone manifest.json for convenience (not source of truth)
        write_standalone_manifest(manifest, run_dir)
        logger.info("Manifest written", path=str(run_dir / "manifest.json"))

    except Exception as e:
        logger.error("Inference failed", error=str(e))
        if close_log_file and log_file_handle is not None:
            log_file_handle.close()
        raise

    logger.info(
        "Inference complete",
        total_judgements=manifest.judgement_count,
        errors=manifest.error_count,
        avg_latency_ms=f"{manifest.avg_latency_ms:.1f}",
    )

    # Close log file if we opened it
    if close_log_file and log_file_handle is not None:
        logger.info("Logs saved", path=str(run_dir / "run.log"))
        log_file_handle.close()
