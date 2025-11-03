"""Ingest-specific I/O configuration schema.

Extends the base IOConfig with ingest-specific fields like dataset_name.
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.schemas.io_config import IOConfig


class IngestIOConfig(IOConfig):
    """Ingest-specific I/O configuration.
    
    Extends base IOConfig with dataset_name field to explicitly identify
    which dataset is being ingested (e.g., 'msmarco', 'trec-covid', 'llmjudge').
    
    This avoids awkward extraction logic and makes the dataset identity explicit.
    
    Example YAML:
        name_hint: llmjudge
        description: LLM Judge Challenge 2024 dataset
        dataset_name: llmjudge
        reader_module: llm_ensemble.ingest.adapters.io.llm_judge_sample_reader
        reader_class: LlmJudgeSampleReader
        writer_module: llm_ensemble.ingest.adapters.io.sql_writer
        writer_class: SqlWriter
    """
    
    dataset_name: str = Field(
        ...,
        description="Dataset identifier (e.g., 'msmarco', 'trec-covid', 'llmjudge')"
    )
