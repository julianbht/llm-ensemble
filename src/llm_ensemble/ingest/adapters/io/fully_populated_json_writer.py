"""Fully populated JSON adapter for judging samples.

Writes judging samples to a single JSON array with all objects fully populated (no references).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.ports import DatasetWriter
from llm_ensemble.libs.utils.entity_filenames import get_entity_filename


class FullyPopulatedJsonWriter(DatasetWriter):
    """Fully populated JSON adapter for judging samples.

    Writes all samples as a single JSON array with full objects embedded.
    Each sample is self-contained with all nested objects fully populated.

    Outputs:
    - run_dir / "ingest_run_info.json" - IngestRunInfo (written once as separate manifest)
    - run_dir / "judging_samples.json" - Samples array (pure domain entities without run_info)

    Example output:
        [
            {"id": "...", "query": {...}, "document": {...}, "gold_score": 2},
            {"id": "...", "query": {...}, "document": {...}, "gold_score": 1}
        ]
    
    Note: run_info is kept separate to maintain clean domain entities and avoid
    duplication. Downstream CLIs can read samples without parsing run_info on each record.
    """

    def write(self, samples: List[JudgingSample], run_dir: Path, run_info: IngestRunInfo) -> WriteSummary:
        """Write fully populated judging samples to a single JSON file.

        Args:
            samples: List of judging samples (pure domain entities)
            run_dir: Run directory where output should be written
            run_info: Immutable runtime context (written to separate manifest)

        Returns:
            WriteSummary tracking write operations (file writes always create all samples)

        Raises:
            IOError: If writing fails
        """
        # Derive filenames from entity class names (DRY principle, following INFER pattern)
        manifest_file = run_dir / get_entity_filename(IngestRunInfo, "json", plural=False)
        samples_file = run_dir / get_entity_filename(JudgingSample, "json")

        # Write run_info manifest (separate from samples)
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(run_info.model_dump(mode="json"), f, indent=2)

        # Write samples file (pure domain entities without run_info)
        samples_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert all samples to JSON-friendly dicts (ensures UUIDs become strings)
        samples_data = [sample.model_dump(mode="json") for sample in samples]

        # Write as a single JSON array
        with samples_file.open("w", encoding="utf-8") as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)

        # File writes always create all samples (no skipping)
        return WriteSummary(samples_created=len(samples))
