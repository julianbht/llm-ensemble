"""Port interface for writing LLM judgements.

Defines the abstract contract for writing judgements to various sinks
(JSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any output format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset


class OutputPort(ABC):
    """Abstract base class for writing LLM judgements with streaming support.

    Implementations can write to different sinks (JSON, Parquet, etc.)
    while providing a consistent interface to the domain service.

    Uses context manager pattern for proper resource lifecycle management.
    Each judgement is written immediately upon calling write_one(), enabling
    fault-tolerant streaming with partial progress preservation.

    The write summary is captured when close() is called and can be retrieved
    after the context manager exits via get_summary().

    Run context (InferRunInfo) is provided once during open() rather than
    being embedded in every judgement, keeping the domain model clean.
    """

    def __init__(self):
        """Initialize writer."""
        self._write_summary: Optional[WriteSummary] = None

    @abstractmethod
    def open(
        self,
        run_dir: Path,
        run_info: InferRunInfo,
        normalized_dataset: NormalizedDataset,
    ) -> OutputPort:
        """Initialize writer with run directory, run context, and input dataset.

        The run_info contains metadata about the inference run (model config,
        git SHA, etc.) including nullable start_idx/end_idx capturing user intent.

        The normalized_dataset is used to link InferRun to the source ingest run for provenance.
        The writer will compute actual start_idx and end_idx from run_info defaults.

        All other metadata (provider, model_config, prompt_template, parser)
        is extracted from the judgement objects during write_one().

        For SQL writers, this creates the InferRun entity and prepares for streaming writes.
        JudgedDataset is created on first write_one() with metadata from the judgement.

        Args:
            run_dir: Run directory where output should be written (writer determines file structure)
            run_info: Inference run context (metadata about model, git state, user-specified indices)
            normalized_dataset: Input dataset being processed (used for provenance linking)

        Returns:
            Self, to enable context manager usage

        Raises:
            IOError: If writer cannot be initialized
        """
        pass

    @abstractmethod
    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to the output sink.

        Must be called within the context manager (after open()).
        Judgement is persisted immediately for fault tolerance.

        Adapter handles its own logging of what entities were written.

        Args:
            judgement: LLMJudgement object to write

        Raises:
            IOError: If write operation fails
            RuntimeError: If called outside of context manager
        """
        pass

    @abstractmethod
    def close(self) -> WriteSummary:
        """Close writer and finalize output.

        Ensures all buffered data is flushed and resources are released.
        Called automatically by context manager __exit__.

        Returns:
            WriteSummary tracking write operations performed during streaming

        Raises:
            IOError: If close operation fails
        """
        pass

    def __enter__(self) -> "OutputPort":
        """Enter context manager.

        Returns:
            Self, for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and ensure cleanup.

        Calls close() and stores the returned WriteSummary for later retrieval.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self._write_summary = self.close()

    def get_summary(self) -> WriteSummary:
        """Get the write summary after the writer has been closed.

        Must be called after the context manager exits.

        Returns:
            WriteSummary tracking write operations

        Raises:
            RuntimeError: If called before writer is closed
        """
        if self._write_summary is None:
            raise RuntimeError("Writer has not been closed yet - must call after context manager exits")
        return self._write_summary
