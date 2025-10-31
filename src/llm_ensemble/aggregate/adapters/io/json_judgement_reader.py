"""JSON adapter for reading LLM judgements.

Reads LLMJudgement records from JSON array files (output from infer CLI).
"""

from __future__ import annotations
import json
from pathlib import Path

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.ports import JudgementReader


class JsonJudgementReader(JudgementReader):
    """Read LLMJudgement records from JSON array files.
    
    Reads from one or more JSON files containing arrays of judgements,
    combining all judgements into a single list for aggregation.
    
    Expected format per file:
        [
            {"judging_sample": {...}, "llm_request": {...}, "llm_response": {...}, ...},
            {"judging_sample": {...}, "llm_request": {...}, "llm_response": {...}, ...}
        ]
    
    Example:
        >>> reader = JsonJudgementReader()
        >>> judgements = reader.read([
        ...     Path("run1/judgements.json"),
        ...     Path("run2/judgements.json"),
        ... ])
        >>> len(judgements)
        200
    """
    
    def read(self, input_paths: list[Path]) -> list[LLMJudgement]:
        """Read LLMJudgement records from one or more JSON array files.
        
        Args:
            input_paths: List of paths to JSON files
            
        Returns:
            List of all LLMJudgement records from all files
            
        Raises:
            FileNotFoundError: If any input file doesn't exist
            ValueError: If JSON is invalid or records don't match schema
        """
        all_judgements: list[LLMJudgement] = []
        
        for input_path in input_paths:
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Read JSON array
            with input_path.open("r", encoding="utf-8") as f:
                try:
                    judgements_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {input_path}: {e}")
            
            # Validate it's an array
            if not isinstance(judgements_data, list):
                raise ValueError(
                    f"Expected JSON array at root of {input_path}, "
                    f"got {type(judgements_data).__name__}"
                )
            
            # Parse each judgement
            for idx, judgement_data in enumerate(judgements_data):
                try:
                    judgement = LLMJudgement(**judgement_data)
                    all_judgements.append(judgement)
                except Exception as e:
                    raise ValueError(
                        f"Invalid judgement at index {idx} in {input_path}: {e}"
                    )
        
        return all_judgements
