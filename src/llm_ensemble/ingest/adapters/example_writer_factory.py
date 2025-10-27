"""Factory for instantiating example writers.

Provides configuration-driven writer selection following the factory pattern.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.ingest.ports import ExampleWriter
from llm_ensemble.ingest.adapters.io import NdjsonExampleWriter


def get_example_writer(output_path: Path, format: str = "ndjson") -> ExampleWriter:
    """Factory function to instantiate example writers.

    Args:
        output_path: Path to output file
        format: Output format identifier (default: 'ndjson')

    Returns:
        ExampleWriter: Concrete writer instance

    Raises:
        ValueError: If format is not recognized

    Example:
        >>> writer = get_example_writer(Path('output.ndjson'))
        >>> writer.write(example)
        >>> writer.close()
    """
    if format == "ndjson":
        return NdjsonExampleWriter(output_path)
    else:
        raise ValueError(
            f"Unknown output format: '{format}'. "
            f"Available formats: ['ndjson']"
        )
