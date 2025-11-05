"""Fully populated NDJSON adapter for judging samples.

Writes judging samples to NDJSON format with all objects fully populated (no references).
"""

from __future__ import annotations
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary
from llm_ensemble.ingest.ports import DatasetWriter


class FullyPopulatedNdjsonWriter(DatasetWriter):
    """Fully populated NDJSON adapter for judging samples.

    Writes each sample as a JSON line with the full manifest object embedded.
    Each sample is self-contained with all nested objects fully populated.

    Output: run_dir / "normalized_dataset.ndjson"

    Example output:
        {"query": {...}, "document": {...}, "gold_score": 2, "manifest": {...}}
        {"query": {...}, "document": {...}, "gold_score": 1, "manifest": {...}}
    """

    def write(self, samples: List[JudgingSample], run_dir: Path) -> WriteSummary:
        """Write fully populated judging samples to NDJSON.

        Args:
            samples: List of judging samples (each contains full manifest)
            run_dir: Run directory where output should be written

        Returns:
            WriteSummary tracking write operations (file writes always create all samples)

        Raises:
            IOError: If writing fails
        """
        # Adapter determines output file structure
        output_path = run_dir / "normalized_dataset.ndjson"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            # Write each judging sample as a JSON line (fully populated)
            for sample in samples:
                json_str = sample.model_dump_json()
                f.write(json_str + "\n")

        # File writes always create all samples (no skipping)
        return WriteSummary(samples_created=len(samples))
