"""Fully populated JSON adapter for reading judging samples.

Reads judging samples from a single JSON array with all objects fully populated.
"""

from __future__ import annotations
import json
from typing import Optional

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.ports import ExampleReader
from llm_ensemble.libs.runtime.path_manager import PathManager


class FullyPopulatedJsonReader(ExampleReader):
    """Fully populated JSON adapter for reading judging samples.

    Reads samples from a single JSON array where each sample has all objects
    embedded (no references). Compatible with output from ingest CLI's
    FullyPopulatedJsonWriter.

    Expected input format:
        [
            {"query": {...}, "document": {...}, "gold_score": 2, "run_info": {...}},
            {"query": {...}, "document": {...}, "gold_score": 1, "run_info": {...}}
        ]
    
    Uses PathManager to resolve run_name to file path, assuming ingest CLI output structure.
    """

    def read(
        self,
        run_name: str,
        limit: Optional[int] = None,
    ) -> list[JudgingSample]:
        """Read judging samples from an ingest run output file.

        Args:
            run_name: Ingest run identifier (e.g., "my_ingest_run")
                     Resolved to artifacts/runs/ingest/{test|official}/{run_name}/judging_samples.json
            limit: Optional maximum number of samples to read

        Returns:
            List of JudgingSample objects

        Raises:
            FileNotFoundError: If run directory or judging_samples.json doesn't exist
            ValueError: If JSON is invalid or samples don't match schema
        """
        # Resolve run_name to file path using PathManager
        input_path = PathManager.get_ingest_output_file(run_name)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as f:
            samples_data = json.load(f)

        if not isinstance(samples_data, list):
            raise ValueError(
                f"Expected JSON array at root, got {type(samples_data).__name__}"
            )

        # Parse samples and apply limit
        samples = [JudgingSample(**sample_data) for sample_data in samples_data]

        if limit is not None:
            samples = samples[:limit]

        return samples
