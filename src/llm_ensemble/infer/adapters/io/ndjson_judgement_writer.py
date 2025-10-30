"""NDJSON adapter for writing LLM judgements.

Writes LLMJudgement records as newline-delimited JSON files.
This is the standard format for downstream aggregate and evaluate CLIs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.ports import JudgementWriter


class NdjsonJudgementWriter(JudgementWriter):
    """Write LLMJudgement records to NDJSON files with streaming support.

    This adapter writes LLMJudgement objects (with sample, llm_response, and manifest)
    as newline-delimited JSON, which is the expected input format for the aggregate CLI.

    Uses context manager pattern for proper file lifecycle. Each judgement is written
    immediately upon calling write_one(), enabling fault-tolerant streaming.

    Example:
        >>> writer = NdjsonJudgementWriter()
        >>> with writer.open(run_dir) as w:
        ...     for judgement in judgements:
        ...         w.write_one(judgement)  # Written immediately to disk
    """

    def __init__(self):
        """Initialize writer."""
        self._file_handle: Optional[TextIO] = None
        self._output_file: Optional[Path] = None

    def open(self, run_dir: Path) -> "NdjsonJudgementWriter":
        """Open NDJSON file for streaming writes.

        Args:
            run_dir: Run directory where output should be written

        Returns:
            Self, to enable context manager usage

        Raises:
            IOError: If file cannot be opened
            RuntimeError: If writer is already open
        """
        if self._file_handle is not None:
            raise RuntimeError("Writer is already open")

        # Writer determines output file structure: judgements.ndjson in run_dir
        self._output_file = run_dir / "judgements.ndjson"

        # Open file for writing (context manager will handle closing)
        self._file_handle = self._output_file.open("w", encoding="utf-8", newline="\n")

        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to NDJSON file.

        Judgement is written immediately and flushed to disk for fault tolerance.

        Args:
            judgement: LLMJudgement object to write

        Raises:
            IOError: If write operation fails
            RuntimeError: If called outside of context manager
        """
        if self._file_handle is None:
            raise RuntimeError("Writer is not open - must call within context manager")

        # Write judgement as JSON line
        json_line = judgement.model_dump_json()
        self._file_handle.write(json_line + "\n")

        # Flush to ensure immediate persistence (fault tolerance)
        self._file_handle.flush()

    def close(self) -> None:
        """Close NDJSON file and release resources.

        Called automatically by context manager __exit__.

        Raises:
            IOError: If close operation fails
        """
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._output_file = None
