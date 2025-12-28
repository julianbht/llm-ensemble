"""Fully populated JSON adapter for judging samples.

Writes judging samples to a single JSON array with all objects fully populated (no references).
Handles its own logging.
"""

from __future__ import annotations
import json

from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig
from llm_ensemble.ingest.application.ports.driven.dataset_writer import DatasetWriter
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.utils.entity_filenames import get_entity_filename
from llm_ensemble.libs.logging.log_events import IngestWriteEvent


class FullyPopulatedJsonWriter(DatasetWriter):
    """Fully populated JSON adapter for judging samples.

    Writes all samples as a single JSON array with full objects embedded.
    Each sample is self-contained with all nested objects fully populated.
    Logs write operations directly.

    Outputs:
    - run_dir / "ingest_run_info.json" - IngestRunInfo (runtime metadata)
    - run_dir / "ingest_run_config.json" - IngestRunConfig (configuration)
    - run_dir / "judging_samples.json" - Samples array (pure domain entities)

    Example output:
        [
            {"id": "...", "query": {...}, "document": {...}, "gold_score": 2},
            {"id": "...", "query": {...}, "document": {...}, "gold_score": 1}
        ]

    Note: run_info and run_config are kept separate to maintain clean domain entities
    and avoid duplication. Downstream CLIs can read samples without parsing metadata on each record.
    """

    def __init__(self):
        """Initialize JSON writer with its own logger."""
        self.logger = get_logger(component="json_writer")

    def write(
        self,
        normalized_dataset: NormalizedDataset,
        run_info: IngestRunInfo,
        run_config: IngestRunConfig,
    ) -> WriteSummary:
        """Write fully populated judging samples to JSON with direct logging.

        Args:
            normalized_dataset: Complete normalized dataset with samples and metadata
            run_info: Immutable runtime metadata (git SHA, timestamps, notes)
            run_config: Immutable run configuration (I/O config, input path, limit)

        Returns:
            WriteSummary as pure data (metadata for run summary)

        Raises:
            IOError: If writing fails
        """
        # Extract samples from normalized dataset
        samples = normalized_dataset.samples

        # Derive run directory from run_info (computed property)
        run_dir = run_info.run_dir

        # Derive filenames from entity class names (DRY principle, following INFER pattern)
        run_info_file = run_dir / get_entity_filename(IngestRunInfo, "json", plural=False)
        run_config_file = run_dir / get_entity_filename(IngestRunConfig, "json", plural=False)
        samples_file = run_dir / get_entity_filename(JudgingSample, "json")

        # Write run_info manifest (runtime metadata)
        with run_info_file.open("w", encoding="utf-8") as f:
            json.dump(run_info.model_dump(mode="json"), f, indent=2)

        # Write run_config manifest (configuration)
        with run_config_file.open("w", encoding="utf-8") as f:
            json.dump(run_config.model_dump(mode="json"), f, indent=2)

        # Write samples file (pure domain entities without run_info)
        samples_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert all samples to JSON-friendly dicts (ensures UUIDs become strings)
        samples_data = [sample.model_dump(mode="json") for sample in samples]

        # Write as a single JSON array
        with samples_file.open("w", encoding="utf-8") as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)

        # Build summary and log (file writes always create all samples, no skipping)
        summary = WriteSummary()
        summary.add_samples(created=len(samples))

        self.logger.info(
            IngestWriteEvent.WRITE_JUDGING_SAMPLES,
            created=len(samples),
            skipped=0,
        )
        self.logger.info(
            IngestWriteEvent.WRITE_COMPLETE,
            total_created=summary.total_created,
            total_skipped=summary.total_skipped,
        )

        return summary
