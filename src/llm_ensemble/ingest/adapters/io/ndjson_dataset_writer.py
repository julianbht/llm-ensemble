"""NDJSON writer for judging samples with manifest.

Writes judging samples to NDJSON format. Each sample contains a reference
to the manifest, establishing a Many-to-One relationship.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample, IngestManifest
from llm_ensemble.ingest.ports import DatasetWriter


class NdjsonDatasetWriter(DatasetWriter):
    """Writer for judging samples in NDJSON format.

    Writes judging samples as a NDJSON file where:
    - First line: Manifest metadata (prefixed with special marker for quick reference)
    - Remaining lines: JudgingSample objects (each includes manifest field), one per line

    This format allows streaming reads while keeping manifest accessible both
    as a separate first line (for quick reference) and embedded in each sample
    (for Many-to-One relationship integrity).

    Example output:
        {"__manifest__": {...}}
        {"query": {...}, "document": {...}, "gold_score": 2, "manifest": {...}}
        {"query": {...}, "document": {...}, "gold_score": 1, "manifest": {...}}
    """

    def write(
        self,
        samples: List[JudgingSample],
        manifest: IngestManifest,
        output_path: Path
    ) -> None:
        """Write judging samples with manifest metadata to NDJSON.

        Args:
            samples: List of judging samples (each contains manifest reference)
            manifest: The ingest manifest for metadata and quick reference
            output_path: Path to output NDJSON file

        Raises:
            IOError: If writing fails
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            # Write manifest as first line with special marker for quick reference
            manifest_line = '{"__manifest__": ' + manifest.model_dump_json() + "}\n"
            f.write(manifest_line)

            # Write each judging sample as a JSON line (includes manifest field)
            for sample in samples:
                json_str = sample.model_dump_json()
                f.write(json_str + "\n")
