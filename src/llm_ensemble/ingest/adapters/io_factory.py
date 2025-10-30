"""Factory for creating I/O adapters for ingest CLI.

Maps I/O config specifications to concrete reader and writer implementations,
enabling dependency injection and configuration-driven adapter selection.

Delegates to IOConfig's built-in instantiation methods, making the config
the single source of truth for adapter selection.
"""

from __future__ import annotations

from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.ingest.ports import SampleReader, DatasetWriter


def get_sample_reader(io_config: IOConfig) -> SampleReader:
    """Create and return the appropriate sample reader adapter.

    Factory function that delegates to IOConfig's get_reader() method,
    which dynamically instantiates the reader from the module path.

    Args:
        io_config: Ingest I/O configuration specifying the reader adapter module path

    Returns:
        SampleReader instance

    Raises:
        ImportError: If the module path cannot be imported

    Example:
        >>> from llm_ensemble.libs.config import load_ingest_io_config
        >>> config = load_ingest_io_config("llm_judge_ingest")
        >>> reader = get_sample_reader(config)
        >>> isinstance(reader, SampleReader)
        True
    """
    return io_config.get_reader()


def get_dataset_writer(io_config: IOConfig) -> DatasetWriter:
    """Create and return the appropriate dataset writer adapter.

    Factory function that delegates to IOConfig's get_writer() method,
    which dynamically instantiates the writer from the module path.

    Args:
        io_config: I/O configuration specifying the writer adapter module path

    Returns:
        DatasetWriter instance

    Raises:
        ImportError: If the module path cannot be imported

    Example:
        >>> from llm_ensemble.libs.config import load_ingest_io_config
        >>> config = load_ingest_io_config("llm_judge_ingest")
        >>> writer = get_dataset_writer(config)
        >>> isinstance(writer, DatasetWriter)
        True
    """
    return io_config.get_writer()
