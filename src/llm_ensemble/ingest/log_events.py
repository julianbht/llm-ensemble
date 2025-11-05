"""Log event enums for ingest CLI.

Centralizes event names for type safety and consistency.
Event values should be descriptive snake_case strings.
"""

from enum import Enum


class IngestLogEvent(str, Enum):
    """Log events for ingest orchestrator."""

    INGEST_STARTED = "ingest_started"
    RUN_DIRECTORY_CREATED = "run_directory_created"
    JUDGING_SAMPLES_BUILT = "judging_samples_built"
    SUMMARY_WRITTEN = "summary_written"
    INGEST_COMPLETE = "ingest_complete"
    INGEST_FAILED = "ingest_failed"
    LOGS_SAVED = "logs_saved"


class IngestWriteEvent(str, Enum):
    """Log events for write operations (storage-agnostic).

    Used by WriteSummary.get_log_entries() to generate structured logs.
    These events work for both database and file-based storage.
    """

    WRITE_DATASETS = "write_datasets"
    WRITE_RUNS = "write_runs"
    WRITE_QUERIES = "write_queries"
    WRITE_DOCUMENTS = "write_documents"
    WRITE_SAMPLES = "write_samples"
    WRITE_COMPLETE = "write_complete"
