"""Fully populated NDJSON adapter for judging samples.

Writes judging samples to NDJSON format with all objects fully populated (no references).
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import json

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.ports import DatasetWriter
from llm_ensemble.libs.utils.entity_filenames import get_entity_filename


class FullyPopulatedNdjsonWriter(DatasetWriter):
    """Fully populated NDJSON adapter for judging samples.

    Writes each sample as a JSON line with the full run_info embedded.
    Each sample is self-contained with all nested objects fully populated.

    Outputs:
    - run_dir / "ingest_run_info.json" - IngestRunInfo (written once)
    - run_dir / "normalized_dataset.ndjson" - Samples (one per line with run_info embedded)

    Example output:
        {"id": "...", "query": {...}, "document": {...}, "gold_score": 2, "run_info": {...}}
        {"id": "...", "query": {...}, "document": {...}, "gold_score": 1, "run_info": {...}}
    """

    def write(self, samples: List[JudgingSample], run_dir: Path, run_info: IngestRunInfo) -> WriteSummary:
        """Write fully populated judging samples to NDJSON.

        Args:
            samples: List of judging samples (pure domain entities)
            run_dir: Run directory where output should be written
            run_info: Immutable runtime context (written to separate manifest and embedded in samples)

        Returns:
            WriteSummary tracking write operations (file writes always create all samples)

        Raises:
            IOError: If writing fails
        """
        # Derive filename from entity class name (following INFER pattern)
        manifest_file = run_dir / get_entity_filename(IngestRunInfo, "json", plural=False)

        # Write run_info manifest (separate from samples)
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(run_info.model_dump(mode="json"), f, indent=2)

        # Write samples file
        output_path = run_dir / "normalized_dataset.ndjson"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            # Write each judging sample as a JSON line with run_info embedded
            for sample in samples:
                # Reconstruct fully populated format at write time (use mode="json" for UUID serialization)
                sample_dict = sample.model_dump(mode="json")
                sample_dict["run_info"] = run_info.model_dump(mode="json")
                json_str = json.dumps(sample_dict)
                f.write(json_str + "\n")

        # File writes always create all samples (no skipping)
        return WriteSummary(samples_created=len(samples))
