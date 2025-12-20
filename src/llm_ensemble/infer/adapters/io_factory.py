"""Builder for IO adapters.

Simple, explicit mapping of IO format names to reader/writer adapter classes.
No decorators, no hidden registration - just a clear dictionary.

IO formats use explicit read→write naming to show the full pipeline:
- db_to_db: Read from PostgreSQL, write to PostgreSQL
- db_to_json: Read from PostgreSQL, write JSON array

To add a new IO format:
1. Create adapter classes that extend ExampleReader and JudgementWriter ports
2. Import them here
3. Add to IO_FORMATS dict with descriptive read→write name
"""

from __future__ import annotations
from typing import Dict, Type, NamedTuple

from llm_ensemble.infer.application.ports.driven.input_port import InputPort
from llm_ensemble.infer.application.ports.driven.output_port import OutputPort
from llm_ensemble.infer.adapters.io.db.db_reader import DBReader
from llm_ensemble.infer.adapters.io.db.db_writer import DBWriter
from llm_ensemble.infer.adapters.io.fully_populated_json_writer import FullyPopulatedJsonWriter


class IOConfig(NamedTuple):
    """Configuration for an IO format."""
    reader_class: Type[InputPort]
    writer_class: Type[OutputPort]
    description: str


# Explicit mapping of IO format names to adapter configurations
IO_FORMATS: Dict[str, IOConfig] = {
    "db_to_db": IOConfig(
        reader_class=DBReader,
        writer_class=DBWriter,
        description="Read from PostgreSQL, write to PostgreSQL"
    ),
    "db_to_json": IOConfig(
        reader_class=DBWriter,
        writer_class=FullyPopulatedJsonWriter,
        description="Read from PostgreSQL, write JSON array"
    ),
}


class IOAdapterFactory:
    """Builder for creating IO adapter instances."""

    @staticmethod
    def create_reader(io_name: str) -> InputPort:
        """Build and return a reader adapter instance.

        Args:
            io_name: Name of the IO format (e.g., 'db_to_json')

        Returns:
            Instantiated reader adapter

        Raises:
            ValueError: If IO format not found
        """
        if io_name not in IO_FORMATS:
            available = ", ".join(sorted(IO_FORMATS.keys()))
            raise ValueError(
                f"IO format '{io_name}' not found. "
                f"Available: {available}"
            )

        config = IO_FORMATS[io_name]
        return config.reader_class(io_name=io_name)

    @staticmethod
    def create_writer(io_name: str) -> OutputPort:
        """Build and return a writer adapter instance.

        Args:
            io_name: Name of the IO format (e.g., 'db_to_json')

        Returns:
            Instantiated writer adapter

        Raises:
            ValueError: If IO format not found
        """
        if io_name not in IO_FORMATS:
            available = ", ".join(sorted(IO_FORMATS.keys()))
            raise ValueError(
                f"IO format '{io_name}' not found. "
                f"Available: {available}"
            )

        config = IO_FORMATS[io_name]
        return config.writer_class(io_name=io_name)

    @staticmethod
    def list_available() -> list[str]:
        """List all available IO format names.

        Returns:
            Sorted list of IO format names
        """
        return sorted(IO_FORMATS.keys())

    @staticmethod
    def has_format(io_name: str) -> bool:
        """Check if IO format is available.

        Args:
            io_name: Name of the IO format

        Returns:
            True if IO format exists
        """
        return io_name in IO_FORMATS

    @staticmethod
    def get_description(io_name: str) -> str:
        """Get description for an IO format.

        Args:
            io_name: Name of the IO format

        Returns:
            Description string

        Raises:
            ValueError: If IO format not found
        """
        if io_name not in IO_FORMATS:
            available = ", ".join(sorted(IO_FORMATS.keys()))
            raise ValueError(
                f"IO format '{io_name}' not found. "
                f"Available: {available}"
            )

        return IO_FORMATS[io_name].description
