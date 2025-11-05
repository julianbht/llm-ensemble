"""Log event enums for ingest CLI.

Centralizes event names for type safety and consistency.
Event values should be descriptive snake_case strings.
"""

from enum import Enum


class IngestLogEvent(str, Enum):
    """Log events for ingest orchestrator."""

    INGEST_STARTED = "ingest_started"
    RUN_DIRECTORY_CREATED = "run_directory_created"
    JUDGING_SAMPLES_PREPARED = "judging_samples_prepared"
    INGEST_SUMMARY_WRITTEN = "ingest_summary_written"
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
    WRITE_JUDGING_SAMPLES = "write_judging_samples"
    WRITE_COMPLETE = "write_complete"


class InferLogEvent(str, Enum):
    """Log events for infer orchestrator."""

    INFER_STARTED = "inference_started"
    RUN_DIRECTORY_CREATED = "run_directory_created"
    SENDING_REQUEST = "sending_request"
    RESPONSE_PARSED = "response_parsed"
    JUDGEMENT_PERSISTED = "judgement_persisted"
    ALL_SAMPLES_PROCESSED = "all_samples_processed"
    INFER_SUMMARY_WRITTEN = "inference_summary_written"
    INFER_COMPLETE = "inference_complete"
    INFER_FAILED = "inference_failed"
    LOGS_SAVED = "logs_saved"
    WARNINGS_COLLECTED = "warnings_collected"


class InferWriteEvent(str, Enum):
    """Log events for judgement write operations.

    Used by WriteSummary.get_log_entries() to generate structured logs.
    """

    WRITE_JUDGEMENT_COMPLETE = "write_judgement_complete"
    WRITE_COMPLETE = "write_complete"
