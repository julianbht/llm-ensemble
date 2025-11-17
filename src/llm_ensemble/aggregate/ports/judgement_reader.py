"""Port interface for judgement readers.

Defines the abstract contract for reading LLMJudgement records from storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement


class JudgementReader(ABC):
    """Abstract base class for reading LLMJudgement records.
    
    Implementations can read from different formats (JSON, Parquet, etc.)
    while providing a consistent interface.
    
    Readers accept run_name strings and internally resolve to file paths
    using PathManager, enabling clean separation of concerns.
    """
    
    @abstractmethod
    def read(self, run_names: list[str]) -> list[LLMJudgement]:
        """Read LLMJudgement records from one or more infer runs.
        
        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])
                      Readers use PathManager to resolve to appropriate file paths
            
        Returns:
            List of all LLMJudgement records from all specified runs
            
        Raises:
            FileNotFoundError: If any run directory or expected files don't exist
            ValueError: If run names are invalid or data is malformed
        """
        pass
