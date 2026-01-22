"""Builder for IO adapters.

IO formats use explicit read→write naming to show the full pipeline:
- db_to_db: Read from PostgreSQL, write to PostgreSQL
"""

from __future__ import annotations

from llm_ensemble.aggregate.application.ports.driven.for_input import ForInput
from llm_ensemble.aggregate.application.ports.driven.for_output import ForOutput
from llm_ensemble.aggregate.adapters.driven.io.db_reader import DBReader
from llm_ensemble.aggregate.adapters.driven.io.db_writer import DBWriter


AVAILABLE_FORMATS = ["db_to_db"]


class IOAdapterFactory:
    """Builder for creating IO adapter instances."""

    @staticmethod
    def create_reader(io_name: str) -> ForInput:
        """Build and return a reader adapter instance.

        Args:
            io_name: Name of the IO format (e.g., 'db_to_db')

        Returns:
            Instantiated reader adapter

        Raises:
            ValueError: If IO format not found
        """
        if io_name == "db_to_db":
            return DBReader()
        else:
            available = ", ".join(sorted(AVAILABLE_FORMATS))
            raise ValueError(
                f"IO format '{io_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def create_writer(io_name: str) -> ForOutput:
        """Build and return a writer adapter instance.

        Args:
            io_name: Name of the IO format (e.g., 'db_to_db')

        Returns:
            Instantiated writer adapter

        Raises:
            ValueError: If IO format not found
        """
        if io_name == "db_to_db":
            return DBWriter(io_name=io_name)
        else:
            available = ", ".join(sorted(AVAILABLE_FORMATS))
            raise ValueError(
                f"IO format '{io_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def list_available() -> list[str]:
        """List all available IO format names."""
        return sorted(AVAILABLE_FORMATS)

    @staticmethod
    def has_format(io_name: str) -> bool:
        """Check if IO format is available."""
        return io_name in AVAILABLE_FORMATS