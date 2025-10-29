"""NDJSON adapter for writing LLM judgements.

Writes LLMJudgement records as newline-delimited JSON files.
This is the standard format for downstream aggregate and evaluate CLIs.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.ports import JudgementWriter


class NdjsonJudgementWriter(JudgementWriter):
    """Write LLMJudgement records to NDJSON files.

    This adapter writes LLMJudgement objects (with sample, llm_response, and manifest)
    as newline-delimited JSON, which is the expected input format for the aggregate CLI.

    Similar to NdjsonDatasetWriter in the ingest pipeline, this writes a batch
    of judgements at once and determines the output file structure.

    Example:
        >>> writer = NdjsonJudgementWriter()
        >>> judgements = [...]  # List of LLMJudgement objects
        >>> writer.write(judgements, run_dir)
    """

    def write(self, judgements: list[LLMJudgement], run_dir: Path) -> None:
        """Write judgements to NDJSON file in run directory.

        Args:
            judgements: List of LLMJudgement objects to write
            run_dir: Run directory where output should be written

        Raises:
            IOError: If write operation fails
        """
        # Writer determines output file structure: judgements.ndjson in run_dir
        output_file = run_dir / "judgements.ndjson"

        # Write all judgements to file
        with output_file.open("w", encoding="utf-8", newline="\n") as f:
            for judgement in judgements:
                json_line = judgement.model_dump_json()
                f.write(json_line + "\n")
