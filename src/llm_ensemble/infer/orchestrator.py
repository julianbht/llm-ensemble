"""Orchestrator for the infer CLI.

This module contains the top-level orchestration logic for running LLM inference
on judging examples. It is separated from the CLI entry point (infer_cli.py)
to enable better testability and reusability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO
import json

from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.infer.config_loaders import load_model_config
from llm_ensemble.infer.providers import iter_judgements
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.logging.logger import get_logger


def _read_examples(input_path: Path) -> list[JudgingExample]:
    """Read NDJSON examples from file."""
    examples = []
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                examples.append(JudgingExample(**json.loads(line)))
    return examples


def _json_dumps(judgement) -> str:
    """Serialize ModelJudgement to JSON."""
    return judgement.model_dump_json()


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
) -> dict:
    """Run LLM inference on judging examples and output structured judgements.
    
    This is the main orchestration function that coordinates:
    - Loading model configuration
    - Setting up run directory and output files
    - Reading input examples
    - Running inference via providers
    - Writing results and manifest
    
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
    
    # Read examples
    logger.info("Reading examples", input_file=str(input_file))
    examples = _read_examples(input_file)
    logger.info("Loaded examples", count=len(examples))
    
    if limit is not None:
        examples = examples[:limit]
        logger.info("Limited examples", count=len(examples))
    
    # Run inference
    count = 0
    error_count = 0
    total_latency_ms = 0.0
    
    try:
        with output_file.open("w", encoding="utf-8", newline="\n") as sink:
            for judgement in iter_judgements(
                iter(examples),
                model_config,
                prompts_dir=prompts_dir,
                prompt_template_name=prompt,
            ):
                sink.write(_json_dumps(judgement) + "\n")
                count += 1
                total_latency_ms += judgement.latency_ms
                
                # Track errors
                if judgement.label is None:
                    error_count += 1
                    logger.warning(
                        "Judgement error",
                        count=count,
                        query_id=judgement.query_id,
                        docid=judgement.docid,
                        warnings=judgement.warnings,
                    )
                else:
                    logger.info(
                        "Processed judgement",
                        count=count,
                        query_id=judgement.query_id,
                        docid=judgement.docid,
                        label=judgement.label,
                        latency_ms=f"{judgement.latency_ms:.1f}",
                    )
    
    except Exception as e:
        logger.error("Inference failed", error=str(e))
        if close_log_file and log_file_handle is not None:
            log_file_handle.close()
        raise
    
    avg_latency = total_latency_ms / count if count > 0 else 0
    logger.info("Inference complete", total_judgements=count, errors=error_count, avg_latency_ms=f"{avg_latency:.1f}")
    
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
            "judgement_count": count,
            "error_count": error_count,
            "avg_latency_ms": avg_latency,
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
        "judgement_count": count,
        "error_count": error_count,
        "avg_latency_ms": avg_latency,
    }
