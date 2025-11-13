"""Fully populated NDJSON adapter for judging samples.

Writes judging samples to NDJSON format with all objects fully populated (no references).
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import json

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary, Dataset
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.ports import DatasetWriter
from llm_ensemble.libs.utils.entity_filenames import get_entity_filename


class FullyPopulatedNdjsonWriter(DatasetWriter):
    """Fully populated NDJSON adapter for judging samples.

    Writes each sample as a JSON line with all nested objects fully populated.
    Each sample is self-contained.

    Outputs:
    - run_dir / "ingest_run_info.json" - IngestRunInfo (written once as separate manifest)
    - run_dir / "judging_samples.ndjson" - Samples (one per line, pure domain entities without run_info)

    Example output:
        {"id": "...", "query": {...}, "document": {...}, "gold_score": 2}
        {"id": "...", "query": {...}, "document": {...}, "gold_score": 1}
    
    Note: run_info is kept separate to maintain clean domain entities and avoid
    duplication. Downstream CLIs can read samples without parsing run_info on each record.
    """

    def write(
        self,
        samples: List[JudgingSample],
        run_info: IngestRunInfo,
        dataset: Dataset,
    ) -> WriteSummary:
        """Write fully populated judging samples to NDJSON.

        Args:
            samples: List of judging samples (pure domain entities)
            run_info: Immutable runtime context (written to separate manifest, contains run_dir)
            dataset: Dataset domain object (not used by file-based writer, for interface compatibility)

        Returns:
            WriteSummary tracking write operations (file writes always create all samples)

        Raises:
            IOError: If writing fails
        """
        # Derive run directory from run_info (computed property)
        run_dir = run_info.run_dir

        # Derive filenames from entity class names (DRY principle, following INFER pattern)
        manifest_file = run_dir / get_entity_filename(IngestRunInfo, "json", plural=False)
        samples_file = run_dir / get_entity_filename(JudgingSample, "ndjson")

        # Write run_info manifest (separate from samples)
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(run_info.model_dump(mode="json"), f, indent=2)

        # Write samples file (pure domain entities without run_info)
        samples_file.parent.mkdir(parents=True, exist_ok=True)

        with samples_file.open("w", encoding="utf-8", newline="\n") as f:
            # Write each judging sample as a JSON line (use mode="json" for UUID serialization)
            for sample in samples:
                json_str = sample.model_dump_json()
                f.write(json_str + "\n")

        # File writes always create all samples (no skipping)
        return WriteSummary(samples_created=len(samples))
