"""Log event enums for ingest CLI.

Centralizes event names for type safety and consistency.
Event values should be descriptive snake_case strings.
"""

from enum import Enum


class IngestLogEvent(str, Enum):
    """Log events for ingest orchestrator."""

    INGEST_STARTED = "ingest_started"
    RUN_DIRECTORY_CREATED = "run_directory_created"
    DATASET_READ_COMPLETE = "dataset_read_complete"
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
    JUDGEMENT_METRICS = "judgement_metrics"
    RETRY_ATTEMPT = "retry_attempt"
    RETRY_EXHAUSTED = "retry_exhausted"
    JUDGEMENT_PERSISTED = "judgement_persisted"
    ALL_SAMPLES_PROCESSED = "all_samples_processed"
    INFER_SUMMARY_WRITTEN = "inference_summary_written"
    INFER_COMPLETE = "inference_complete"
    INFER_FAILED = "inference_failed"
    LOGS_SAVED = "logs_saved"
    WARNINGS_COLLECTED = "warnings_collected"


class InferWriteEvent(str, Enum):
    """Log events for judgement write operations."""

    # Run metadata entities (written once during open)
    WRITE_PROVIDERS = "write_providers"
    WRITE_MODEL_SPECS = "write_model_specs"
    WRITE_PROMPT_TEMPLATES = "write_prompt_templates"
    WRITE_PARSER_SPECS = "write_parser_specs"
    WRITE_INFER_RUNS = "write_infer_runs"

    # Per-judgement entities (written during write_one)
    WRITE_LLM_REQUESTS = "write_llm_requests"
    WRITE_LLM_SCORES = "write_llm_scores"
    WRITE_LLM_CALLS = "write_llm_calls"

    # Dataset finalization (written during close)
    WRITE_JUDGED_DATASETS = "write_judged_datasets"
    WRITE_JUDGED_DATASET_JUNCTIONS = "write_judged_dataset_junctions"

    # Final summary
    WRITE_COMPLETE = "write_complete"
