"""Port interface for aggregated judgement writers.

Defines the abstract contract for writing AggregatedJudgement records to storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from llm_ensemble.aggregate.schemas import AggregatedJudgement


class AggregatedJudgementWriter(ABC):
    """Abstract base class for writing AggregatedJudgement records.
    
    Implementations can write to different formats (JSON, Parquet, etc.)
    while providing a consistent interface.
    
    Uses context manager pattern for proper resource management.
    """
    
    @abstractmethod
    def open(self, run_dir: Path) -> "AggregatedJudgementWriter":
        """Open writer for streaming writes.
        
        Args:
            run_dir: Run directory where output should be written
            
        Returns:
            Self, to enable context manager usage
            
        Raises:
            IOError: If file cannot be opened
            RuntimeError: If writer is already open
        """
        pass
    
    @abstractmethod
    def write_one(self, aggregated_judgement: AggregatedJudgement) -> None:
        """Write a single aggregated judgement.
        
        Args:
            aggregated_judgement: AggregatedJudgement to write
            
        Raises:
            IOError: If write operation fails
            RuntimeError: If called outside of context manager
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close writer and release resources.
        
        Called automatically by context manager __exit__.
        """
        pass
    
    def __enter__(self) -> "AggregatedJudgementWriter":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
