"""NDJSON adapter for reading judging examples.

Reads newline-delimited JSON files containing JudgingSample records.
This is the standard format produced by the ingest CLI.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.ports import ExampleReader


class NdjsonExampleReader(ExampleReader):
    """Read JudgingSample records from NDJSON files.

    This adapter reads the output format produced by the ingest CLI:
    one JudgingSample JSON object per line.

    Example:
        >>> reader = NdjsonExampleReader()
        >>> examples = reader.read(Path("samples.ndjson"), limit=100)
        >>> len(examples)
        100
    """

    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> list[JudgingSample]:
        """Read examples from NDJSON file.

        Args:
            input_path: Path to NDJSON file with JudgingSample records
            limit: Optional maximum number of examples to read

        Returns:
            List of JudgingSample objects

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If JSON parsing fails
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        examples = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    example_dict = json.loads(line)
                    examples.append(JudgingSample(**example_dict))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {input_path}: {e}"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid JudgingSample on line {line_num} in {input_path}: {e}"
                    )

                # Apply limit if specified
                if limit is not None and len(examples) >= limit:
                    break

        return examples
