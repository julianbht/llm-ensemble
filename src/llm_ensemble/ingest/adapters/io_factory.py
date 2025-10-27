"""Factory for creating I/O adapters for ingest CLI.

Maps I/O config specifications to concrete reader and writer implementations,
enabling dependency injection and configuration-driven adapter selection.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.infer.schemas import IOConfig
from llm_ensemble.ingest.ports import ExampleReader, ExampleWriter
from llm_ensemble.ingest.adapters.io import (
    LlmJudgeExampleReader,
    NdjsonExampleWriter,
)


def get_example_reader(io_config: IOConfig, dataset_id: str) -> ExampleReader:
    """Create and return the appropriate example reader adapter.

    Factory function that instantiates the correct reader implementation
    based on the I/O configuration's reader field.

    Args:
        io_config: I/O configuration specifying the reader adapter
        dataset_id: Dataset identifier to embed in JudgingExample records

    Returns:
        ExampleReader instance

    Raises:
        ValueError: If reader adapter is not supported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_io_config
        >>> config = load_io_config("llm_judge_ingest")
        >>> reader = get_example_reader(config, "llm-judge-2024")
        >>> isinstance(reader, LlmJudgeExampleReader)
        True
    """
    reader_name = io_config.reader.lower()

    if reader_name == "llm_judge_example_reader":
        return LlmJudgeExampleReader(dataset_id=dataset_id)
    else:
        raise ValueError(
            f"Unsupported example reader: {io_config.reader}. "
            f"Supported readers: llm_judge_example_reader"
        )


def get_example_writer(io_config: IOConfig, output_path: Path) -> ExampleWriter:
    """Create and return the appropriate example writer adapter.

    Factory function that instantiates the correct writer implementation
    based on the I/O configuration's writer field.

    Args:
        io_config: I/O configuration specifying the writer adapter
        output_path: Path where examples should be written

    Returns:
        ExampleWriter instance

    Raises:
        ValueError: If writer adapter is not supported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_io_config
        >>> config = load_io_config("llm_judge_ingest")
        >>> writer = get_example_writer(config, Path("out.ndjson"))
        >>> isinstance(writer, NdjsonExampleWriter)
        True
    """
    writer_name = io_config.writer.lower()

    if writer_name == "ndjson_example_writer":
        return NdjsonExampleWriter(output_path)
    else:
        raise ValueError(
            f"Unsupported example writer: {io_config.writer}. "
            f"Supported writers: ndjson_example_writer"
        )
