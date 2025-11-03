"""Ingest-specific I/O configuration schema.

Extends the base IOConfig with ingest-specific fields like dataset_name and dataset_description.
"""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.schemas.io_config import IOConfig


class IngestIOConfig(IOConfig):
    """Ingest-specific I/O configuration.
    
    Extends base IOConfig with dataset_name and dataset_description fields to explicitly
    identify which dataset is being ingested (e.g., 'msmarco', 'trec-covid', 'llmjudge').
    Used to create the Dataset entity and compute UUIDs of queries, documents, and JudgingSamples.
    
    Example YAML:
        name_hint: llmjudge
        description: LLM Judge Challenge 2024 dataset
        dataset_name: llmjudge
        dataset_description: LLM Judge Challenge 2024 - Information Retrieval Benchmark
        reader_module: llm_ensemble.ingest.adapters.io.llm_judge_sample_reader
        reader_class: LlmJudgeSampleReader
        writer_module: llm_ensemble.ingest.adapters.io.sql_writer
        writer_class: SqlWriter
    """
    
    dataset_name: str = Field(
        ...,
        description="Dataset identifier (e.g., 'msmarco', 'trec-covid', 'llmjudge')"
    )
    
    dataset_description: Optional[str] = Field(
        None,
        description="Optional human-readable description of the dataset"
    )
