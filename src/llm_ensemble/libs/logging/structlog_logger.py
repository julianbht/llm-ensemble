"""Structlog-based logging for LLM Ensemble CLIs.

This module provides a structlog-based logger that supports both human-readable
pretty printing and structured JSON output, configured via LoggingConfig.
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Optional, Any
import structlog


def _drop_stdlib_fields(_, __, event_dict):
    """Remove stdlib integration fields from output."""
    # Remove stdlib integration fields that leak into output
    event_dict.pop("_from_structlog", None)
    event_dict.pop("_record", None)
    return event_dict


def configure_logger(
    cli_name: str,
    run_name: Optional[str] = None,
    pretty_print: bool = True,
    save_logs: bool = False,
    log_file_path: Optional[Path] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> structlog.stdlib.BoundLogger:
    """Configure structlog for a CLI run with flexible formatting options.

    Args:
        cli_name: Name of the CLI (e.g., "ingest", "infer", "aggregate", "evaluate")
        run_name: Optional run ID for context
        pretty_print: If True, use human-readable console output with colors.
                     If False, use structured JSON output.
        save_logs: If True, save logs to file (requires log_file_path)
        log_file_path: Optional path to write logs to (required if save_logs=True)
        console_level: Minimum log level for console output (DEBUG, INFO, WARNING, ERROR)
        file_level: Minimum log level for file output (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured structlog logger with bound context

    Raises:
        ValueError: If save_logs=True but log_file_path is None

    Example:
        >>> logger = configure_logger(
        ...     cli_name="infer",
        ...     run_name="20250115_143022_phi3",
        ...     pretty_print=True,
        ...     save_logs=True,
        ...     log_file_path=Path("artifacts/runs/infer/20250115_143022_phi3/run.log"),
        ... )
        >>> logger.info("inference_started", model="phi3-mini", num_samples=100)
    """
    # Validate log file path if save_logs is True
    if save_logs and log_file_path is None:
        raise ValueError("log_file_path must be provided when save_logs=True")

    # Suppress noisy third-party library logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Shared processors for all outputs
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Console renderer - always human-readable one-line format
    console_renderer = structlog.dev.ConsoleRenderer(
        colors=False,
        exception_formatter=structlog.dev.plain_traceback,
    )
    # Add processor to drop stdlib fields before rendering
    console_processors = shared_processors + [_drop_stdlib_fields, console_renderer]

    # File renderer - JSON format (pretty or compact based on pretty_print setting)
    if pretty_print:
        # Pretty-printed JSON with indentation for readability
        file_renderer = structlog.processors.JSONRenderer(indent=2, sort_keys=True)
    else:
        # Compact JSON (one line per log entry)
        file_renderer = structlog.processors.JSONRenderer()

    # File processors also need to drop stdlib fields
    file_processors = shared_processors + [_drop_stdlib_fields, file_renderer]

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up stdlib logging handlers
    handlers = []

    # Console handler (stderr)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(console_level.upper())
    stderr_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=console_processors,
        )
    )
    handlers.append(stderr_handler)

    # File handler (if save_logs=True)
    if save_logs and log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(file_level.upper())
        # Use JSON for file output (pretty or compact based on setting)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=file_processors,
            )
        )
        handlers.append(file_handler)

    # Configure root logger with handlers
    logging.basicConfig(
        format="%(message)s",
        level=min(
            getattr(logging, console_level.upper()),
            getattr(logging, file_level.upper()) if save_logs else logging.CRITICAL
        ),
        handlers=handlers,
        force=True,  # Override any existing config
    )

    # Create logger without binding context (cli_name and run_name are not needed)
    logger = structlog.get_logger()

    return logger
