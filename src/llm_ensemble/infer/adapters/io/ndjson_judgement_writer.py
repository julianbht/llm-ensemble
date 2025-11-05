"""NDJSON adapter for writing LLM judgements.

Writes LLMJudgement records as newline-delimited JSON files.
This is the standard format for downstream aggregate and evaluate CLIs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.libs.schemas.write_result import WriteResult


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
        super().__init__()
        self._file_handle: Optional[TextIO] = None
        self._output_file: Optional[Path] = None
        self._judgements_written: int = 0

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

        # Reset counter for new write session
        self._judgements_written = 0

        return self

    def write_one(self, judgement: LLMJudgement) -> WriteResult:
        """Write a single judgement to NDJSON file.

        Judgement is written immediately and flushed to disk for fault tolerance.

        Args:
            judgement: LLMJudgement object to write

        Returns:
            WriteResult for this specific write operation (contains judgement ID)

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

        # Track write
        self._judgements_written += 1

        # Return result for this specific write (per-operation feedback)
        return WriteResult(
            item_id=judgement.judging_sample.id,
            item_type="judgement"
        )

    def close(self) -> WriteSummary:
        """Close NDJSON file and release resources.

        Called automatically by context manager __exit__.

        Returns:
            WriteSummary with aggregate statistics (total judgements written across entire run)

        Raises:
            IOError: If close operation fails
        """
        # Create aggregate summary (total across all writes)
        summary = WriteSummary(judgements_written=self._judgements_written)

        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._output_file = None

        return summary
