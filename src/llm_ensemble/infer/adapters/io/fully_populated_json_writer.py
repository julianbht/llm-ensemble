"""Fully populated JSON adapter for writing LLM judgements.

Writes judgements to a single JSON array with streaming support.
Accumulates judgements in memory and writes all at once on close.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.ports import JudgementWriter


class FullyPopulatedJsonWriter(JudgementWriter):
    """Fully populated JSON adapter for writing LLM judgements.

    Accumulates judgements in memory and writes them as a single JSON array
    when the writer is closed. All objects are fully populated (no references).

    Output: run_dir / "judgements.json"

    Example output:
        [
            {"judging_sample": {...}, "llm_response": {...}, "llm_score": {...}, "run_info": {...}},
            {"judging_sample": {...}, "llm_response": {...}, "llm_score": {...}, "run_info": {...}}
        ]

    Note: This adapter accumulates all judgements in memory before writing.
    For very large datasets, consider using the NDJSON adapter instead.
    """

    def __init__(self):
        """Initialize the JSON writer."""
        self.output_path: Optional[Path] = None
        self.judgements: list[LLMJudgement] = []

    def open(self, run_dir: Path) -> "FullyPopulatedJsonWriter":
        """Initialize writer and prepare for streaming.

        Args:
            run_dir: Run directory where output should be written

        Returns:
            Self, to enable context manager usage

        Raises:
            RuntimeError: If writer is already open
        """
        if self.output_path is not None:
            raise RuntimeError("Writer is already open")

        self.output_path = run_dir / "judgements.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.judgements = []

        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Accumulate a single judgement in memory.

        Judgement will be written to disk when close() is called.

        Args:
            judgement: LLMJudgement object to write

        Raises:
            RuntimeError: If called outside of context manager
        """
        if self.output_path is None:
            raise RuntimeError("Writer must be opened before writing")

        self.judgements.append(judgement)

    def close(self) -> None:
        """Write all accumulated judgements to a single JSON file.

        Raises:
            IOError: If write operation fails
        """
        if self.output_path is None:
            return  # Already closed or never opened

        # Convert all judgements to dicts
        judgements_data = [judgement.model_dump() for judgement in self.judgements]

        # Write as a single JSON array
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(judgements_data, f, indent=2, ensure_ascii=False)

        # Reset state
        self.output_path = None
        self.judgements = []
