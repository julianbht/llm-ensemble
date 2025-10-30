"""Factory for creating I/O adapters based on configuration.

Maps I/O config specifications to concrete reader and writer implementations,
enabling dependency injection and loose coupling.

Delegates to IOConfig's built-in instantiation methods, making the config
the single source of truth for adapter selection.
"""

from __future__ import annotations

from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.infer.ports import ExampleReader, JudgementWriter


def get_example_reader(io_config: IOConfig) -> ExampleReader:
    """Create and return the appropriate example reader adapter.

    Factory function that delegates to IOConfig's get_reader() method,
    which dynamically instantiates the reader from the module path.

    Args:
        io_config: I/O configuration specifying the reader adapter module path

    Returns:
        ExampleReader instance

    Raises:
        ImportError: If the module path cannot be imported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_io_config
        >>> config = load_io_config("ndjson")
        >>> reader = get_example_reader(config)
        >>> isinstance(reader, ExampleReader)
        True
    """
    return io_config.get_reader()


def get_judgement_writer(io_config: IOConfig) -> JudgementWriter:
    """Create and return the appropriate judgement writer adapter.

    Factory function that delegates to IOConfig's get_writer() method,
    which dynamically instantiates the writer from the module path.

    Args:
        io_config: I/O configuration specifying the writer adapter module path

    Returns:
        JudgementWriter instance

    Raises:
        ImportError: If the module path cannot be imported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_io_config
        >>> config = load_io_config("ndjson")
        >>> writer = get_judgement_writer(config)
        >>> isinstance(writer, JudgementWriter)
        True
    """
    return io_config.get_writer()
