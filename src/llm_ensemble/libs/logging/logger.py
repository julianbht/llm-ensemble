"""Human-readable logging for CLI applications.

Provides a dual-mode logger that supports both human-readable terminal output
and structured JSON logging (via LOG_FORMAT=json environment variable).
"""

from __future__ import annotations
import sys
import json
import os
from datetime import datetime
from typing import Any, Optional
from enum import Enum


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Logger:
    """Simple human-readable logger with optional JSON output.

    Features:
    - Color-coded terminal output (human mode)
    - Structured JSON output (JSON mode via LOG_FORMAT=json)
    - Contextual logging with CLI name and run_id
    - Timestamps on every log

    Example:
        >>> logger = Logger(cli_name="ingest", run_id="20250115_143022_llm-judge")
        >>> logger.info("Processing example", query_id="q1", docid="d1")
        [2025-01-15 14:30:22] INFO [ingest:20250115_143022_llm-judge] Processing example query_id=q1 docid=d1
    """

    # ANSI color codes
    COLORS = {
        LogLevel.DEBUG: "\033[36m",     # Cyan
        LogLevel.INFO: "\033[32m",      # Green
        LogLevel.WARNING: "\033[33m",   # Yellow
        LogLevel.ERROR: "\033[31m",     # Red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(
        self,
        cli_name: str,
        run_id: Optional[str] = None,
        use_json: Optional[bool] = None,
        log_file: Optional[Any] = None,
        console_level: LogLevel = LogLevel.INFO,
        file_level: LogLevel = LogLevel.DEBUG,
    ):
        """Initialize logger.

        Args:
            cli_name: Name of the CLI (e.g., "ingest", "infer")
            run_id: Optional run ID for context
            use_json: Force JSON output (defaults to LOG_FORMAT env var)
            log_file: Optional file handle to write logs to (in addition to stderr)
            console_level: Minimum log level for console output (default: INFO)
            file_level: Minimum log level for file output (default: DEBUG)
        """
        self.cli_name = cli_name
        self.run_id = run_id
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level

        # Determine output format
        if use_json is None:
            log_format = os.getenv("LOG_FORMAT", "human").lower()
            self.use_json = log_format == "json"
        else:
            self.use_json = use_json

        # Detect if stderr is a TTY for color support
        self.use_color = sys.stderr.isatty() and not self.use_json

    def _should_log_console(self, level: LogLevel) -> bool:
        """Check if this log level should be output to console."""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
        return levels.index(level) >= levels.index(self.console_level)

    def _should_log_file(self, level: LogLevel) -> bool:
        """Check if this log level should be output to file."""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
        return levels.index(level) >= levels.index(self.file_level)

    def _format_human(self, level: LogLevel, message: str, use_color: bool = None, **kwargs: Any) -> str:
        """Format log message for human-readable output.

        Args:
            level: Log level
            message: Log message
            use_color: Override color setting (defaults to self.use_color)
            **kwargs: Additional key-value pairs to append
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build context (just cli_name, skip run_id to save space)
        context = self.cli_name

        # Apply color if requested
        should_use_color = use_color if use_color is not None else self.use_color
        if should_use_color:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level.value}{self.RESET}"
            context_str = f"{self.BOLD}[{context}]{self.RESET}"
        else:
            level_str = level.value
            context_str = f"[{context}]"

        # Build kwargs string with smart formatting
        kwargs_str = ""
        if kwargs:
            # Separate short and long fields for better readability
            short_fields = []
            long_fields = []

            for k, v in kwargs.items():
                v_str = str(v)
                # Fields like query, doc, prompt, raw_response are typically long
                if k in ['query', 'doc', 'prompt', 'raw_response'] or len(v_str) > 100:
                    long_fields.append((k, v_str))
                else:
                    short_fields.append(f"{k}={v_str}")

            # Short fields on same line
            if short_fields:
                kwargs_str = " " + " ".join(short_fields)

            # Long fields on separate lines with indentation
            if long_fields:
                for k, v in long_fields:
                    kwargs_str += f"\n  {k}: {v}"

        return f"[{timestamp}] {level_str} {context_str} {message}{kwargs_str}"

    def _format_json(self, level: LogLevel, message: str, **kwargs: Any) -> str:
        """Format log message as JSON."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "cli_name": self.cli_name,
            "message": message,
            **kwargs,
        }
        if self.run_id:
            log_entry["run_id"] = self.run_id

        return json.dumps(log_entry)

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Internal logging method."""
        should_log_console = self._should_log_console(level)
        should_log_file = self.log_file is not None and self._should_log_file(level)

        if not should_log_console and not should_log_file:
            return

        # Format messages
        if self.use_json:
            formatted_message = self._format_json(level, message, **kwargs)
            console_output = formatted_message
            file_output = formatted_message
        else:
            console_output = self._format_human(level, message, use_color=self.use_color, **kwargs)
            file_output = self._format_human(level, message, use_color=False, **kwargs)

        # Write to console if level qualifies
        if should_log_console:
            print(console_output, file=sys.stderr, flush=True)

        # Write to file if level qualifies
        if should_log_file:
            print(file_output, file=self.log_file, flush=True)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)


def get_logger(
    cli_name: str,
    run_id: Optional[str] = None,
    log_file: Optional[Any] = None,
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
) -> Logger:
    """Get a logger instance for a CLI.

    Args:
        cli_name: Name of the CLI (e.g., "ingest", "infer")
        run_id: Optional run ID for context
        log_file: Optional file handle to write logs to (in addition to stderr)
        console_level: Minimum log level for console (default: INFO)
        file_level: Minimum log level for file (default: DEBUG)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("ingest", run_id="20250115_143022_llm-judge")
        >>> logger.info("Starting ingest")

        >>> # With file logging (DEBUG logs only go to file)
        >>> with open("run.log", "w") as f:
        >>>     logger = get_logger("infer", run_id="abc123", log_file=f)
        >>>     logger.debug("Detailed info")  # Only in file
        >>>     logger.info("Processing...")   # Console + file
    """
    return Logger(
        cli_name=cli_name,
        run_id=run_id,
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
    )
