"""JSON adapter for reading LLM judgements.

Reads LLMJudgement records from JSON array files (output from infer CLI).
"""

from __future__ import annotations
import json

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.ports import JudgementReader
from llm_ensemble.libs.runtime.path_manager import PathManager


class JsonJudgementReader(JudgementReader):
    """Read LLMJudgement records from JSON array files.
    
    Reads from one or more infer run outputs, combining all judgements 
    into a single list for aggregation.
    
    Uses PathManager to resolve run names to file paths, assuming infer CLI output structure.
    """
    
    def read(self, run_names: list[str]) -> list[LLMJudgement]:
        """Read LLMJudgement records from one or more infer runs.
        
        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])
                      Resolved to artifacts/runs/infer/{test|official}/{run_name}/llm_judgements.json
            
        Returns:
            List of all LLMJudgement records from all runs
            
        Raises:
            FileNotFoundError: If any run directory or llm_judgements.json doesn't exist
            ValueError: If JSON is invalid or records don't match schema
        """
        all_judgements: list[LLMJudgement] = []
        
        for run_name in run_names:
            # Resolve run_name to file path using PathManager
            input_path = PathManager.get_infer_output_file(run_name)
            
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
