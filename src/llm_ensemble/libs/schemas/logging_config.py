"""Logging configuration schema for structlog-based logging.

This module defines the Pydantic schema for logging configuration,
controlling pretty printing and log file saving behavior.
"""

from __future__ import annotations
from pydantic import ConfigDict, Field

from llm_ensemble.libs.schemas.base_config import BaseConfig
from llm_ensemble.libs.runtime.path_manager import PathManager


class LoggingConfig(BaseConfig):
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

    Example YAML:
        ```yaml
        name_hint: pretty
        pretty_print: true
        save_logs: true
        console_level: INFO
        file_level: DEBUG
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

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load(cls, config_name: str) -> LoggingConfig:
        """Load and validate logging configuration from YAML file.

        Args:
            config_name: Name of the logging config file (without .yaml extension)
                        e.g., "standard", "json", "console-only"

        Returns:
            Validated LoggingConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        return super().load(
            config_name=config_name,
            config_dir=PathManager.get_configs_dir() / "logging"
        )
