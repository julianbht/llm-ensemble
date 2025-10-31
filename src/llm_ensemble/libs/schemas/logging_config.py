"""Logging configuration schema for structlog-based logging.

This module defines the Pydantic schema for logging configuration,
controlling pretty printing and log file saving behavior.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Configuration for structlog logging behavior.

    Controls both console output formatting (pretty vs JSON) and whether logs
    should be saved to a file in the run directory.

    Attributes:
        pretty_print: If True, use human-readable console output with colors.
                     If False, use structured JSON output to console.
        save_logs: If True, save logs to run.log file in the run directory.
                  If False, logs only go to console.
        console_level: Minimum log level for console output (DEBUG, INFO, WARNING, ERROR).
        file_level: Minimum log level for file output (DEBUG, INFO, WARNING, ERROR).
        name_hint: Short identifier for this config (used in run IDs).

    Example YAML:
        ```yaml
        pretty_print: true
        save_logs: true
        console_level: INFO
        file_level: DEBUG
        name_hint: pretty
        ```
    """

    pretty_print: bool = Field(
        default=True,
        description="Use human-readable console output with colors (true) or structured JSON (false)"
    )

    save_logs: bool = Field(
        default=True,
        description="Save logs to run.log file in run directory"
    )

    console_level: str = Field(
        default="INFO",
        description="Minimum log level for console output (DEBUG, INFO, WARNING, ERROR)",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$"
    )

    file_level: str = Field(
        default="DEBUG",
        description="Minimum log level for file output (DEBUG, INFO, WARNING, ERROR)",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$"
    )

    name_hint: str = Field(
        default="default",
        description="Short identifier for this config (used in run IDs)"
    )

    class Config:
        """Pydantic config."""
        extra = "forbid"  # Reject unknown fields
