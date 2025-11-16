"""Port interface for judgement readers.

Defines the abstract contract for reading LLMJudgement records from storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement


class JudgementReader(ABC):
    """Abstract base class for reading LLMJudgement records.
    
    Implementations can read from different formats (JSON, Parquet, etc.)
    while providing a consistent interface.
    """
    
    @abstractmethod
    def read(self, input_paths: list[Path]) -> list[LLMJudgement]:
        """Read LLMJudgement records from one or more input files.
        
        Args:
            input_paths: List of paths to files containing LLMJudgement records
            
        Returns:
            List of all LLMJudgement records from all input files
            
        Raises:
            FileNotFoundError: If any input file doesn't exist
            ValueError: If data is invalid or doesn't match schema
        """
        pass
