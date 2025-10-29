"""Port interface for writing LLM judgements.

Defines the abstract contract for writing judgements to various sinks
(NDJSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any output format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement


class JudgementWriter(ABC):
    """Abstract base class for writing LLM judgements.

    Implementations can write to different sinks (NDJSON, Parquet, etc.)
    while providing a consistent interface to the domain service.

    Similar to DatasetWriter in the ingest pipeline, this writes a batch
    of judgements at once (writer determines output file structure).

    Example:
        >>> writer = NdjsonJudgementWriter(output_file)
        >>> judgements = [...]  # List of LLMJudgement objects
        >>> writer.write(judgements, run_dir)
    """

    @abstractmethod
    def write(self, judgements: list[LLMJudgement], run_dir: Path) -> None:
        """Write judgements to output sink.

        Args:
            judgements: List of LLMJudgement objects to write
            run_dir: Run directory where output should be written (writer determines file structure)

        Raises:
            IOError: If write operation fails
        """
        pass
