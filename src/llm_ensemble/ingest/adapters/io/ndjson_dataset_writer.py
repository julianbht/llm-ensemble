"""NDJSON writer for NormalizedDataset.

Writes a complete NormalizedDataset (samples + manifest) to NDJSON format.
The output file contains all judging samples as NDJSON, with the manifest
embedded as metadata.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.ingest.schemas import NormalizedDataset
from llm_ensemble.ingest.ports import DatasetWriter


class NdjsonDatasetWriter(DatasetWriter):
    """Writer for NormalizedDataset in NDJSON format.

    Writes the complete dataset as a single NDJSON file where:
    - First line: Manifest metadata (prefixed with special marker)
    - Remaining lines: JudgingSample objects, one per line

    This format allows streaming reads while keeping manifest bundled with data.

    Example output:
        {"__manifest__": {...}}
        {"query": {...}, "document": {...}, "gold_score": 2}
        {"query": {...}, "document": {...}, "gold_score": 1}
    """

    def write(self, dataset: NormalizedDataset, output_path: Path) -> None:
        """Write a complete NormalizedDataset to NDJSON.

        Args:
            dataset: The NormalizedDataset to write (samples + manifest)
            output_path: Path to output NDJSON file

        Raises:
            IOError: If writing fails
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            # Write manifest as first line with special marker
            manifest_line = '{"__manifest__": ' + dataset.manifest.model_dump_json() + "}\n"
            f.write(manifest_line)

            # Write each judging sample as a JSON line
            for sample in dataset.judging_samples:
                json_str = sample.model_dump_json()
                f.write(json_str + "\n")
