"""Factory for creating I/O adapter instances.

Explicit instantiation of I/O adapters with format-specific constructors.
Each adapter defines its own constructor signature and configuration needs.

To add a new I/O format:
1. Create adapter classes that extend ForInput/ForOutput
2. Import them here
3. Add explicit instantiation case in create_reader/create_writer methods
"""

from __future__ import annotations

from llm_ensemble.evaluate.adapters.driven.io.dummy_writer import DummyWriter
from llm_ensemble.evaluate.application.ports.driven.for_input import ForInput
from llm_ensemble.evaluate.application.ports.driven.for_output import ForOutput
from llm_ensemble.evaluate.adapters.driven.io.db_infer_reader import DBInferReader
from llm_ensemble.evaluate.adapters.driven.io.db_aggregate_reader import DBAggregateReader


AVAILABLE_FORMATS = ["dummy", "db_infer", "db_aggregate"]


class IOAdapterFactory:
    """Factory for creating I/O adapter instances."""

    @staticmethod
    def create_reader(io_name: str) -> ForInput:
        """Build and return a reader adapter instance.

        Args:
            io_name: Name of the I/O format (e.g., 'db_to_html')

        Returns:
            Instantiated reader adapter

        Raises:
            ValueError: If I/O format not found
        """
        if io_name == "db_infer":
            return DBInferReader(io_name=io_name)
        elif io_name == "db_aggregate":
            return DBAggregateReader(io_name=io_name)
        else:
            available = ", ".join(sorted(AVAILABLE_FORMATS))
            raise ValueError(
                f"I/O format '{io_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def create_writer(io_name: str) -> ForOutput:
        """Build and return a writer adapter instance.

        Args:
            io_name: Name of the I/O format (e.g., 'db_to_html')

        Returns:
            Instantiated writer adapter

        Raises:
            ValueError: If I/O format not found
        """
        # For now, all formats use dummy writer
        # Future: implement HTML, JSON, etc. writers
        return DummyWriter(io_name=io_name)

    @staticmethod
    def list_available() -> list[str]:
        """List all available I/O format names."""
        return sorted(AVAILABLE_FORMATS)

    @staticmethod
    def has_format(io_name: str) -> bool:
        """Check if I/O format is available."""
        return io_name in AVAILABLE_FORMATS
