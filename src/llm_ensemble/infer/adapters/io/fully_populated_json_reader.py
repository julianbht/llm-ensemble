"""Fully populated JSON adapter for reading judging samples.

Reads judging samples from a single JSON array with all objects fully populated.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.ports import ExampleReader


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
    """

    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> list[JudgingSample]:
        """Read judging samples from a JSON array file.

        Args:
            input_path: Path to JSON file containing array of samples
            limit: Optional maximum number of samples to read

        Returns:
            List of JudgingSample objects

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If JSON is invalid or samples don't match schema
        """
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
