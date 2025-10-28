"""Factory for creating I/O adapters based on configuration.

Maps I/O config specifications to concrete reader and writer implementations,
enabling dependency injection and loose coupling.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.infer.schemas import IOConfig
from llm_ensemble.infer.ports import ExampleReader, JudgementWriter
from llm_ensemble.infer.adapters.io import (
    NdjsonExampleReader,
    NdjsonJudgementWriter,
)


def get_example_reader(io_config: IOConfig) -> ExampleReader:
    """Create and return the appropriate example reader adapter.

    Factory function that instantiates the correct reader implementation
    based on the I/O configuration's reader field.

    Args:
        io_config: I/O configuration specifying the reader adapter

    Returns:
        ExampleReader instance

    Raises:
        ValueError: If reader adapter is not supported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_io_config
        >>> config = load_io_config("ndjson")
        >>> reader = get_example_reader(config)
        >>> isinstance(reader, NdjsonExampleReader)
        True
    """
    reader_name = io_config.reader.lower()

    if reader_name == "ndjson_example_reader":
        return NdjsonExampleReader()
    else:
        raise ValueError(
            f"Unsupported example reader: {io_config.reader}. "
            f"Supported readers: ndjson_example_reader"
        )


def get_judgement_writer(io_config: IOConfig, output_path: Path) -> JudgementWriter:
    """Create and return the appropriate judgement writer adapter.

    Factory function that instantiates the correct writer implementation
    based on the I/O configuration's writer field.

    Args:
        io_config: I/O configuration specifying the writer adapter
        output_path: Path where judgements should be written

    Returns:
        JudgementWriter instance

    Raises:
        ValueError: If writer adapter is not supported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_io_config
        >>> config = load_io_config("ndjson")
        >>> writer = get_judgement_writer(config, Path("out.ndjson"))
        >>> isinstance(writer, NdjsonJudgementWriter)
        True
    """
    writer_name = io_config.writer.lower()

    if writer_name == "ndjson_judgement_writer":
        return NdjsonJudgementWriter(output_path)
    else:
        raise ValueError(
            f"Unsupported judgement writer: {io_config.writer}. "
            f"Supported writers: ndjson_judgement_writer"
        )
