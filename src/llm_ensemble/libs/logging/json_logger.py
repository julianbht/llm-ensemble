"""Structured JSON logging configuration for all CLIs.

Provides consistent structlog setup with JSON output for machine-readable logs.
Logs are written to both stderr (for real-time visibility) and file (for analysis).
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


def configure_logging(
    cli_name: str,
    run_id: str,
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    git_sha: Optional[str] = None,
) -> structlog.stdlib.BoundLogger:
    """Configure structured JSON logging for a CLI run.

    Sets up structlog to output JSON logs to both stderr and an optional file.
    All log records are automatically enriched with cli, run_id, and git_sha context.

    Args:
        cli_name: Name of the CLI (e.g., "ingest", "infer", "aggregate", "evaluate")
        run_id: Unique run identifier
        log_file: Optional path to write logs to (in addition to stderr)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        git_sha: Optional git SHA for reproducibility tracking

    Returns:
        Configured structlog logger with bound context

    Example:
        >>> logger = configure_logging(
        ...     cli_name="infer",
        ...     run_id="20250115_143022_phi3",
        ...     log_file=Path("artifacts/runs/infer/20250115_143022_phi3/logs.jsonl"),
        ...     git_sha="abc123def"
        ... )
        >>> logger.info("inference_started", model="phi3-mini", num_samples=100)
        >>> logger.error("inference_failed", query_id="q1", error="timeout")
    """
    # Configure shared processors for structlog
    shared_processors = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add timestamp in ISO format
        structlog.processors.TimeStamper(fmt="iso"),
        # Format exception info
        structlog.processors.format_exc_info,
    ]

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            # Use ProcessorFormatter for rendering
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up stdlib logging
    # Stderr handler (always JSON)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    )

    # Configure root logger
    handlers = [stderr_handler]

    # File handler (if log_file provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        )
        handlers.append(file_handler)

    # Configure root logger with handlers
    logging.basicConfig(
        format="%(message)s",
        level=log_level.upper(),
        handlers=handlers,
        force=True,  # Override any existing config
    )

    # Create logger with bound context
    logger = structlog.get_logger(cli_name)

    # Bind shared context to all log records
    context = {
        "cli": cli_name,
        "run_id": run_id,
    }
    if git_sha:
        context["git_sha"] = git_sha

    return logger.bind(**context)
