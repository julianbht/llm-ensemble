"""JSON adapter for writing aggregated judgements.

Writes AggregatedJudgement records as a JSON array file.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from llm_ensemble.aggregate.schemas import AggregatedJudgement
from llm_ensemble.aggregate.ports import AggregatedJudgementWriter


class JsonAggregatedJudgementWriter(AggregatedJudgementWriter):
    """Write AggregatedJudgement records to a JSON array file.
    
    Buffers all judgements in memory and writes them as a single JSON array
    when the writer is closed. Simple and human-readable.
    """
    
    def __init__(self):
        """Initialize writer."""
        self._output_file: Optional[Path] = None
        self._buffer: list[AggregatedJudgement] = []
        self._is_open: bool = False
    
    def open(self, run_dir: Path) -> "JsonAggregatedJudgementWriter":
        """Open writer for buffered writes.
        
        Args:
            run_dir: Run directory where output should be written
            
        Returns:
            Self, to enable context manager usage
            
        Raises:
            RuntimeError: If writer is already open
        """
        if self._is_open:
            raise RuntimeError("Writer is already open")
        
        # Writer determines output file structure: aggregated.json in run_dir
        self._output_file = run_dir / "aggregated.json"
        self._buffer = []
        self._is_open = True
        
        return self
    
    def write_one(self, aggregated_judgement: AggregatedJudgement) -> None:
        """Buffer a single aggregated judgement for writing.
        
        Judgements are buffered in memory and written as a JSON array on close.
        
        Args:
            aggregated_judgement: AggregatedJudgement object to write
            
        Raises:
            RuntimeError: If called outside of context manager
        """
        if not self._is_open:
            raise RuntimeError("Writer is not open - must call within context manager")
        
        self._buffer.append(aggregated_judgement)
    
    def close(self) -> None:
        """Write buffered judgements as JSON array and release resources.
        
        Called automatically by context manager __exit__.
        
        Raises:
            IOError: If write operation fails
        """
        if self._is_open:
            # Convert all buffered judgements to dict
            judgements_data = [j.model_dump() for j in self._buffer]
            
            # Write as JSON array
            with self._output_file.open("w", encoding="utf-8") as f:
                json.dump(judgements_data, f, indent=2)
            
            # Clean up
            self._buffer = []
            self._output_file = None
            self._is_open = False
