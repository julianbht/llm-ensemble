"""Structlog-based logging for LLM Ensemble CLIs.

This module provides a structlog-based logger that supports both human-readable
pretty printing and structured JSON output, configured via environment variables.
"""

from __future__ import annotations
import os
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import structlog


def _convert_enums_to_values(_, __, event_dict):
    """Convert all Enum values to their string values automatically.

    This allows using enums directly in log calls without needing to call .value:
        logger.info(MyEvent.SOME_EVENT, key=value)  # instead of MyEvent.SOME_EVENT.value
    """
    # Convert the event name if it's an enum
    if isinstance(event_dict.get("event"), Enum):
        event_dict["event"] = event_dict["event"].value

    # Convert any enum values in the event_dict
    for key, value in list(event_dict.items()):
        if isinstance(value, Enum):
            event_dict[key] = value.value

    return event_dict


def _drop_stdlib_fields(_, __, event_dict):
    """Remove stdlib integration fields from output."""
    # Remove stdlib integration fields that leak into output
    event_dict.pop("_from_structlog", None)
    event_dict.pop("_record", None)
    return event_dict


class CustomConsoleRenderer:
    """Custom console renderer for cleaner component display.

    Formats log entries with component as a bracket prefix instead of key=value.
    Example: [sql_writer] write_datasets created=1 skipped=0
    """

    def __init__(self):
        """Initialize the custom console renderer."""
        self._pad_event = 30  # Padding for event names

    def __call__(self, _, __, event_dict):
        """Render log entry with custom formatting.

        Args:
            event_dict: Dictionary containing log event data

        Returns:
            Formatted log string
        """
        # Extract and remove standard fields that we don't want to display
        level = event_dict.pop("level", "info")
        event = event_dict.pop("event", "")
        event_dict.pop("timestamp", None)  # Remove timestamp from output
        event_dict.pop("component", None)  # Remove component from output

        # Remove observability metadata from console (keep minimal)
        event_dict.pop("cli", None)
        event_dict.pop("run_name", None)
        event_dict.pop("run_type", None)

        # Build output parts
        parts = []

        # Level in brackets
        parts.append(f"[{level:8s}]")

        # Event name (padded for alignment)
        parts.append(f"{event:{self._pad_event}s}")

        # Remaining key-value pairs
        for key, value in event_dict.items():
            parts.append(f"{key}={value}")

        return " ".join(parts)


def _get_logging_config_from_env() -> dict[str, Any]:
    """Read logging configuration from environment variables.

    Environment variables:
        LOG_PRETTY_PRINT: Use human-readable console output (true/false, default: false)
        LOG_SAVE_LOGS: Save logs to file (true/false, default: true)
        LOG_CONSOLE_LEVEL: Console log level (DEBUG/INFO/WARNING/ERROR, default: INFO)
        LOG_FILE_LEVEL: File log level (DEBUG/INFO/WARNING/ERROR, default: DEBUG)

    Returns:
        Dictionary with logging configuration values
    """
    def parse_bool(value: str, default: bool) -> bool:
        """Parse boolean from string with default."""
        if not value:
            return default
        return value.lower() in ("true", "1", "yes")

    return {
        "pretty_print": parse_bool(os.getenv("LOG_PRETTY_PRINT", ""), default=False),
        "save_logs": parse_bool(os.getenv("LOG_SAVE_LOGS", ""), default=True),
        "console_level": os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper(),
        "file_level": os.getenv("LOG_FILE_LEVEL", "DEBUG").upper(),
    }


def configure_logger(
    cli_name: str,
    run_name: Optional[str] = None,
    run_type: Optional[str] = None,
    log_file_path: Optional[Path] = None,
) -> structlog.stdlib.BoundLogger:
    """Configure structlog for a CLI run using environment variables.

    Configuration is read from environment variables:
        LOG_PRETTY_PRINT: Use human-readable console output (true/false, default: false)
        LOG_SAVE_LOGS: Save logs to file (true/false, default: true)
        LOG_CONSOLE_LEVEL: Console log level (DEBUG/INFO/WARNING/ERROR, default: INFO)
        LOG_FILE_LEVEL: File log level (DEBUG/INFO/WARNING/ERROR, default: DEBUG)

    Args:
        cli_name: Name of the CLI (e.g., "ingest", "infer", "aggregate", "evaluate")
        run_name: Optional run ID for context
        run_type: Optional run type ("test" or "official")
        log_file_path: Optional path to write logs to (required if LOG_SAVE_LOGS=true)

    Returns:
        Configured structlog logger with bound context

    Raises:
        ValueError: If LOG_SAVE_LOGS=true but log_file_path is None
    """
    # Read logging configuration from environment
    config = _get_logging_config_from_env()
    pretty_print = config["pretty_print"]
    save_logs = config["save_logs"]
    console_level = config["console_level"]
    file_level = config["file_level"]

    # Validate log file path if save_logs is True
    if save_logs and log_file_path is None:
        raise ValueError("log_file_path must be provided when LOG_SAVE_LOGS=true")

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
        _convert_enums_to_values,  # Convert enums to their string values automatically
    ]

    # Console renderer - custom renderer for cleaner component display
    console_renderer = CustomConsoleRenderer()
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

    # Bind observability context to contextvars (process-wide)
    # This makes the context available to all loggers in the process
    # These fields will appear in log files but are stripped from console output
    bound_context = {"cli": cli_name}
    if run_name:
        bound_context["run_name"] = run_name
    if run_type:
        bound_context["run_type"] = run_type
    
    structlog.contextvars.bind_contextvars(**bound_context)
    
    # Return a logger (context is now global via contextvars)
    return structlog.get_logger()


def get_logger(component: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a logger with optional component binding.
    
    This should be called AFTER configure_logger() has been called by the orchestrator.
    The logger will automatically include CLI context (cli, run_name, run_type) from
    contextvars, plus any component name you provide.
    
    Args:
        component: Optional component name to bind to the logger (e.g., "sql_writer")
    
    Returns:
        Configured structlog logger with CLI context and optional component binding
        
    Example:
        # In orchestrator
        configure_logger(cli_name="infer", run_name="test_run")
        
        # In adapter/service
        logger = get_logger(component="sql_writer")
        logger.info("write_datasets", created=5, skipped=2)
        # Output includes: cli=infer, run_name=test_run, component=sql_writer
    """
    logger = structlog.get_logger()
    if component:
        logger = logger.bind(component=component)
    return logger


def clear_logging_context() -> None:
    """Clear the global logging context.
    
    This should be called at the end of a CLI run to prevent context leaking
    between runs (especially important for testing).
    """
    structlog.contextvars.clear_contextvars()
