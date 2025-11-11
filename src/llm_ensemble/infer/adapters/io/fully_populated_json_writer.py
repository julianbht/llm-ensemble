"""Fully populated JSON adapter for writing LLM judgements.

Writes judgements to a single JSON array with streaming support.
Writes run metadata as a separate manifest JSON file.
Accumulates judgements in memory and writes all at once on close.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.libs.schemas.write_result import WriteResult


class FullyPopulatedJsonWriter(JudgementWriter):
    """Fully populated JSON adapter for writing LLM judgements.

    Accumulates judgements in memory and writes them as a single JSON array
    when the writer is closed. Run metadata is written to a separate manifest file.

    Output files:
    - run_dir / "judgements.json" - Array of judgements
    - run_dir / "run_manifest.json" - Run metadata

    Example judgements.json:
        [
            {"judging_sample": {...}, "prompt": "...", "llm_response": {...}, "llm_score": {...}},
            {"judging_sample": {...}, "prompt": "...", "llm_response": {...}, "llm_score": {...}}
        ]

    Note: This adapter accumulates all judgements in memory before writing.
    For very large datasets, consider using the NDJSON adapter instead.
    """

    def __init__(self):
        """Initialize the JSON writer."""
        super().__init__()
        self.output_path: Optional[Path] = None
        self.manifest_path: Optional[Path] = None
        self.judgements: list[LLMJudgement] = []

    def open(self, run_dir: Path, run_info: InferRunInfo) -> "FullyPopulatedJsonWriter":
        """Initialize writer, write manifest, and prepare for streaming.

        Args:
            run_dir: Run directory where output should be written
            run_info: Inference run context (written to separate manifest file)

        Returns:
            Self, to enable context manager usage

        Raises:
            RuntimeError: If writer is already open
        """
        if self.output_path is not None:
            raise RuntimeError("Writer is already open")

        self.output_path = run_dir / "judgements.json"
        self.manifest_path = run_dir / "run_manifest.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.judgements = []

        # Write run manifest immediately (separate from judgements)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(run_info.model_dump(mode="json"), f, indent=2)

        return self

    def write_one(self, judgement: LLMJudgement) -> WriteResult:
        """Accumulate a single judgement in memory.

        Judgement will be written to disk when close() is called.

        Args:
            judgement: LLMJudgement object to write

        Returns:
            WriteResult for this specific write operation (contains judgement ID)

        Raises:
            RuntimeError: If called outside of context manager
        """
        if self.output_path is None:
            raise RuntimeError("Writer must be opened before writing")

        self.judgements.append(judgement)

        # Return result for this specific write (per-operation feedback)
        return WriteResult(
            item_id=judgement.judging_sample.id,
            item_type="judgement"
        )

    def close(self) -> WriteSummary:
        """Write all accumulated judgements to a single JSON file.

        Returns:
            WriteSummary tracking number of judgements written

        Raises:
            IOError: If write operation fails
        """
        judgements_count = len(self.judgements)

        if self.output_path is not None:
            # Convert to JSON-ready dicts so UUIDs and datetimes serialize correctly
            judgements_data = [judgement.model_dump(mode="json") for judgement in self.judgements]

            # Write as a single JSON array
            with self.output_path.open("w", encoding="utf-8") as f:
                json.dump(judgements_data, f, indent=2, ensure_ascii=False)

            # Reset state
            self.output_path = None
            self.judgements = []

        return WriteSummary(judgements_written=judgements_count)
