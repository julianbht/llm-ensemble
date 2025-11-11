"""NDJSON adapter for writing LLM judgements.

Writes LLMJudgement records as newline-delimited JSON files.
Writes run metadata as a separate manifest JSON file.
This is the standard format for downstream aggregate and evaluate CLIs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO
import json

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.libs.schemas.write_result import WriteResult
from llm_ensemble.libs.utils.entity_filenames import get_entity_filename


class NdjsonJudgementWriter(JudgementWriter):
    """Write LLMJudgement records to NDJSON files with streaming support.

    This adapter writes:
    - LLMJudgement objects to llm_judgements.ndjson (one per line)
    - InferRunInfo to infer_run_info.json (separate file, written once)

    Filenames are derived from entity class names using get_entity_filename().
    This separation keeps run metadata out of individual judgements, reducing
    duplication and file size. The manifest format is standard for downstream CLIs.

    Uses context manager pattern for proper file lifecycle. Each judgement is written
    immediately upon calling write_one(), enabling fault-tolerant streaming.

    Example:
        >>> writer = NdjsonJudgementWriter()
        >>> with writer.open(run_dir, run_info) as w:
        ...     for judgement in judgements:
        ...         w.write_one(judgement)  # Written immediately to disk
    """

    def __init__(self):
        """Initialize writer."""
        super().__init__()
        self._file_handle: Optional[TextIO] = None
        self._output_file: Optional[Path] = None
        self._manifest_file: Optional[Path] = None
        self._judgements_written: int = 0

    def open(self, run_dir: Path, run_info: InferRunInfo) -> "NdjsonJudgementWriter":
        """Open NDJSON file for streaming writes and write manifest.

        Args:
            run_dir: Run directory where output should be written
            run_info: Inference run context (written to separate manifest file)

        Returns:
            Self, to enable context manager usage

        Raises:
            IOError: If file cannot be opened
            RuntimeError: If writer is already open
        """
        if self._file_handle is not None:
            raise RuntimeError("Writer is already open")

        # Derive filenames from entity class names (DRY principle)
        self._output_file = run_dir / get_entity_filename(LLMJudgement, "ndjson")
        self._manifest_file = run_dir / get_entity_filename(InferRunInfo, "json", plural=False)

        # Write run manifest immediately (separate from judgements)
        with self._manifest_file.open("w", encoding="utf-8") as f:
            json.dump(run_info.model_dump(mode="json"), f, indent=2)

        # Open judgements file for streaming writes
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
            self._manifest_file = None

        return summary
