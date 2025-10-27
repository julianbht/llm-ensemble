"""Factory for creating I/O adapters for ingest CLI.

Maps I/O config specifications to concrete reader and writer implementations,
enabling dependency injection and configuration-driven adapter selection.
"""

from __future__ import annotations

from llm_ensemble.ingest.schemas import IngestIOConfig
from llm_ensemble.ingest.ports import SampleReader, DatasetWriter
from llm_ensemble.ingest.adapters.io import (
    LlmJudgeSampleReader,
    NdjsonDatasetWriter,
)


def get_sample_reader(io_config: IngestIOConfig) -> SampleReader:
    """Create and return the appropriate sample reader adapter.

    Factory function that instantiates the correct reader implementation
    based on the I/O configuration's reader field.

    Args:
        io_config: Ingest I/O configuration specifying the reader adapter

    Returns:
        SampleReader instance

    Raises:
        ValueError: If reader adapter is not supported

    Example:
        >>> from llm_ensemble.libs.config import load_ingest_io_config
        >>> config = load_ingest_io_config("llm_judge_ingest")
        >>> reader = get_sample_reader(config)
        >>> isinstance(reader, LlmJudgeSampleReader)
        True
    """
    reader_name = io_config.reader.lower()

    if reader_name == "llm_judge_sample_reader":
        return LlmJudgeSampleReader()
    else:
        raise ValueError(
            f"Unsupported sample reader: {io_config.reader}. "
            f"Supported readers: llm_judge_sample_reader"
        )


def get_dataset_writer(io_config: IngestIOConfig) -> DatasetWriter:
    """Create and return the appropriate dataset writer adapter.

    Factory function that instantiates the correct writer implementation
    based on the I/O configuration's writer field.

    Args:
        io_config: I/O configuration specifying the writer adapter

    Returns:
        DatasetWriter instance

    Raises:
        ValueError: If writer adapter is not supported

    Example:
        >>> from llm_ensemble.libs.config import load_ingest_io_config
        >>> config = load_ingest_io_config("llm_judge_ingest")
        >>> writer = get_dataset_writer(config)
        >>> isinstance(writer, NdjsonDatasetWriter)
        True
    """
    writer_name = io_config.writer.lower()

    if writer_name == "ndjson_dataset_writer":
        return NdjsonDatasetWriter()
    else:
        raise ValueError(
            f"Unsupported dataset writer: {io_config.writer}. "
            f"Supported writers: ndjson_dataset_writer"
        )
