"""Builder for IO adapters.

Explicit instantiation of IO adapters with format-specific constructors.
Each adapter defines its own constructor signature and configuration needs.

To add a new IO format:
1. Create adapter classes that extend ForInput/ForOutput
2. Import them here
3. Add explicit instantiation case in create_reader/create_writer methods
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.ingest.application.ports.driven.for_input import ForInput
from llm_ensemble.ingest.application.ports.driven.for_output import ForOutput
from llm_ensemble.ingest.adapters.driven.io.llm_judge_dataset_reader import LlmJudgeDatasetReader
from llm_ensemble.ingest.adapters.driven.io.db.db_writer import DbWriter


AVAILABLE_FORMATS = ["llm_judge_ingest"]


class IOAdapterFactory:
    """Builder for creating IO adapter instances."""

    @staticmethod
    def create_reader(io_name: str) -> ForInput:
        """Build and return a reader adapter instance.

        Args:
            io_name: Name of the IO format (e.g., 'llm_judge_ingest')

        Returns:
            Instantiated reader adapter

        Raises:
            ValueError: If IO format not found
        """
        if io_name == "llm_judge_ingest":
            return LlmJudgeDatasetReader(io_name=io_name)
        else:
            available = ", ".join(sorted(AVAILABLE_FORMATS))
            raise ValueError(
                f"IO format '{io_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def create_writer(io_name: str, run_dir: Path) -> ForOutput:
        """Build and return a writer adapter instance.

        Args:
            io_name: Name of the IO format (e.g., 'llm_judge_ingest')
            run_dir: Run directory for file-based writers

        Returns:
            Instantiated writer adapter

        Raises:
            ValueError: If IO format not found
        """
        if io_name == "llm_judge_ingest":
            return DbWriter(io_name=io_name, run_dir=run_dir)
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
