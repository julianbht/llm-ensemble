"""Domain layer for the ingest CLI.

This package contains pure business logic with no infrastructure dependencies.
All I/O operations are abstracted via ports.
"""

from llm_ensemble.ingest.domain.ingestion_service import IngestionService

__all__ = ["IngestionService"]
