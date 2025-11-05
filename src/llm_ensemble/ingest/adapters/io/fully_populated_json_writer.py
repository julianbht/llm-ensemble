"""Fully populated JSON adapter for judging samples.

Writes judging samples to a single JSON array with all objects fully populated (no references).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary
from llm_ensemble.ingest.ports import DatasetWriter


class FullyPopulatedJsonWriter(DatasetWriter):
    """Fully populated JSON adapter for judging samples.

    Writes all samples as a single JSON array with full objects embedded.
    Each sample is self-contained with all nested objects fully populated.

    Output: run_dir / "normalized_dataset.json"

    Example output:
        [
            {"query": {...}, "document": {...}, "gold_score": 2, "run_info": {...}},
            {"query": {...}, "document": {...}, "gold_score": 1, "run_info": {...}}
        ]
    """

    def write(self, samples: List[JudgingSample], run_dir: Path) -> WriteSummary:
        """Write fully populated judging samples to a single JSON file.

        Args:
            samples: List of judging samples (each contains full run_info)
            run_dir: Run directory where output should be written

        Returns:
            WriteSummary tracking write operations (file writes always create all samples)

        Raises:
            IOError: If writing fails
        """
        # Adapter determines output file structure
        output_path = run_dir / "normalized_dataset.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert all samples to JSON-friendly dicts (ensures UUIDs become strings)
        samples_data = [sample.model_dump(mode="json") for sample in samples]

        # Write as a single JSON array
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)

        # File writes always create all samples (no skipping)
        return WriteSummary(samples_created=len(samples))
