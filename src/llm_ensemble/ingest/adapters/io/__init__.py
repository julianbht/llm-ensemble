"""I/O adapters for the ingest CLI.

This package contains concrete implementations of the ExampleWriter port
for different output formats.
"""

from llm_ensemble.ingest.adapters.io.ndjson_example_writer import NdjsonExampleWriter

__all__ = ["NdjsonExampleWriter"]
