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


def configure_logger(
    cli_name: str,
    run_id: Optional[str] = None,
    pretty_print: bool = True,
    save_logs: bool = False,
    log_file_path: Optional[Path] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> structlog.stdlib.BoundLogger:
    """Configure structlog for a CLI run with flexible formatting options.

    Args:
        cli_name: Name of the CLI (e.g., "ingest", "infer", "aggregate", "evaluate")
        run_id: Optional run ID for context
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
        ...     run_id="20250115_143022_phi3",
        ...     pretty_print=True,
        ...     save_logs=True,
        ...     log_file_path=Path("artifacts/runs/infer/20250115_143022_phi3/run.log"),
        ... )
        >>> logger.info("inference_started", model="phi3-mini", num_samples=100)
    """
    # Validate log file path if save_logs is True
    if save_logs and log_file_path is None:
        raise ValueError("log_file_path must be provided when save_logs=True")

    # Shared processors for all outputs
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Console renderer (pretty or JSON)
    if pretty_print:
        console_renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),  # Only use colors if outputting to terminal
        )
    else:
        console_renderer = structlog.processors.JSONRenderer()

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
            processor=console_renderer,
        )
    )
    handlers.append(stderr_handler)

    # File handler (if save_logs=True)
    if save_logs and log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(file_level.upper())
        # Always use JSON for file output for structured logs
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processor=structlog.processors.JSONRenderer(),
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

    # Create logger with bound context
    logger = structlog.get_logger(cli_name)

    # Bind shared context to all log records
    context: dict[str, Any] = {"cli": cli_name}
    if run_id:
        context["run_id"] = run_id

    return logger.bind(**context)
