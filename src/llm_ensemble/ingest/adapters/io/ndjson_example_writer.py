"""NDJSON writer for JudgingExample records.

Writes JudgingExample records to newline-delimited JSON (NDJSON) format,
with one JSON object per line.
"""

from __future__ import annotations
from pathlib import Path
from typing import TextIO, Optional

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.ingest.ports import ExampleWriter


class NdjsonExampleWriter(ExampleWriter):
    """Writer for JudgingExample records in NDJSON format.

    Each JudgingExample is serialized to JSON and written on a single line,
    following the newline-delimited JSON (NDJSON) format.

    Example:
        >>> writer = NdjsonExampleWriter(Path("output.ndjson"))
        >>> for example in examples:
        ...     writer.write(example)
        >>> writer.close()
    """

    def __init__(self, output_path: Path):
        """Initialize NDJSON writer.

        Args:
            output_path: Path to output NDJSON file
        """
        self.output_path = output_path
        self._file: Optional[TextIO] = None
        self._opened = False

    def _ensure_open(self) -> None:
        """Open output file if not already open."""
        if not self._opened:
            self._file = self.output_path.open("w", encoding="utf-8", newline="\n")
            self._opened = True

    def write(self, example: JudgingExample) -> None:
        """Write a single JudgingExample as a JSON line.

        Args:
            example: The JudgingExample to write

        Raises:
            IOError: If writing fails
        """
        self._ensure_open()
        if self._file is None:
            raise IOError("Failed to open output file")

        # Serialize to JSON and write with newline
        json_str = example.model_dump_json()
        self._file.write(json_str + "\n")

    def close(self) -> None:
        """Close the writer and flush any buffered data.

        Raises:
            IOError: If flushing/closing fails
        """
        if self._file is not None:
            self._file.close()
            self._file = None
            self._opened = False
