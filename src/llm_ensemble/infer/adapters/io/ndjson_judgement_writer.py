"""NDJSON adapter for writing model judgements.

Writes ModelJudgement records as newline-delimited JSON files.
This is the standard format for downstream aggregate and evaluate CLIs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

from llm_ensemble.infer.schemas import ModelJudgement
from llm_ensemble.infer.ports import JudgementWriter


class NdjsonJudgementWriter(JudgementWriter):
    """Write ModelJudgement records to NDJSON files.

    This adapter writes one ModelJudgement JSON object per line,
    which is the expected input format for the aggregate CLI.

    Example:
        >>> writer = NdjsonJudgementWriter(Path("judgements.ndjson"))
        >>> writer.write(judgement1)
        >>> writer.write(judgement2)
        >>> writer.close()

        Or using context manager:
        >>> with NdjsonJudgementWriter(Path("judgements.ndjson")) as writer:
        ...     writer.write(judgement)
    """

    def __init__(self, output_path: Path):
        """Initialize writer with output path.

        Args:
            output_path: Path to NDJSON file to write
        """
        self.output_path = output_path
        self._file_handle: Optional[TextIO] = None
        self._closed = False

    def write(self, judgement: ModelJudgement) -> None:
        """Write a single judgement to NDJSON file.

        Args:
            judgement: ModelJudgement object to write

        Raises:
            IOError: If file is closed or write fails
        """
        if self._closed:
            raise IOError("Cannot write to closed writer")

        # Lazy open file on first write
        if self._file_handle is None:
            self._file_handle = self.output_path.open(
                "w", encoding="utf-8", newline="\n"
            )

        # Serialize and write
        json_line = judgement.model_dump_json()
        self._file_handle.write(json_line + "\n")

    def close(self) -> None:
        """Finalize and close the output file.

        This method is idempotent - safe to call multiple times.
        """
        if not self._closed and self._file_handle is not None:
            self._file_handle.close()
            self._closed = True
