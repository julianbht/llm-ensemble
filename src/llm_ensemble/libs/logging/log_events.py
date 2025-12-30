"""Log event enums for ingest CLI.

Centralizes event names for type safety and consistency.
Event values should be descriptive snake_case strings.
"""

from enum import Enum


class IngestLogEvent(str, Enum):
    """Log events for ingest orchestrator."""

    INGEST_STARTED = "ingest_started"
    READ_COMPLETE = "dataset_read_complete"
    INGEST_SUMMARY_WRITTEN = "ingest_summary_written"
    LOGS_SAVED = "logs_saved"
    PERSISTENCE_COMPLETE = "persistence_complete"


class IngestWriteEvent(str, Enum):
    """Log events for write operations (storage-agnostic).

    Used by WriteSummary.get_log_entries() to generate structured logs.
    These events work for both database and file-based storage.
    """

    WRITE_QUERIES = "write_queries"
    WRITE_DOCUMENTS = "write_documents"
    WRITE_JUDGING_SAMPLES = "write_judging_samples"
    WRITE_NORMALIZED_DATASET = "write_normalized_dataset"
    WRITE_DATASET_SAMPLES = "write_dataset_samples"
    WRITE_RUNS = "write_runs"
    WRITE_COMPLETE = "write_complete"
    WRITE_RUN_CONFIG = "write_run_config"


class InferLogEvent(str, Enum):
    """Log events for infer orchestrator."""

    INFER_STARTED = "inference_started"
    SENDING_REQUEST = "sending_request"
    RESPONSE_PARSED = "response_parsed"
    AGREEMENT_CHECKED = "agreement_checked"
    COST_CALCULATED = "cost_calculated"
    TOKENS_COUNTED = "tokens_counted"
    RETRY_ATTEMPT = "retry_attempt"
    RETRY_EXHAUSTED = "retry_exhausted"
    ALL_SAMPLES_PROCESSED = "all_samples_processed"
    INFER_SUMMARY_WRITTEN = "inference_summary_written"
    INFER_COMPLETE = "inference_complete"
    LOGS_SAVED = "logs_saved"
    WARNINGS_COLLECTED = "warnings_collected"
    BUILDING_PROMPT = "building_prompt"
    PARSING_REQUEST = "parsing_request"


class InferWriteEvent(str, Enum):
    """Log events for judgement write operations."""

    # Run metadata entities (written once during open)
    WRITE_PROVIDERS = "write_providers"
    WRITE_MODELS = "write_models"
    WRITE_MODEL_CONFIGS = "write_model_configs"
    WRITE_PARSERS = "write_parsers"
    WRITE_PROMPT_TEMPLATES = "write_prompt_templates"
    WRITE_INFER_RUN_CONFIGS = "write_infer_run_configs"
    WRITE_INFER_RUNS = "write_infer_runs"

    # Per-judgement entities (written during write_one)
    WRITE_LLM_PROMPTS = "write_llm_prompts"
    WRITE_LLM_RESPONSES = "write_llm_responses"
    WRITE_LLM_INVOCATION_METRICS = "write_llm_invocation_metrics"
    WRITE_LLM_SCORES = "write_llm_scores"
    WRITE_LLM_JUDGEMENTS = "write_llm_judgements"

    # Dataset finalization (written during close)
    WRITE_INFER_RUN_OUTPUTS = "write_infer_run_outputs"
    WRITE_JUDGED_DATASETS = "write_judged_datasets"
    WRITE_JUDGED_DATASET_JUNCTIONS = "write_judged_dataset_junctions"

    # Final summary
    WRITE_COMPLETE = "write_complete"


class AggregateLogEvent(str, Enum):
    """Log events for aggregate orchestrator."""

    AGGREGATE_STARTED = "aggregation_started"
    DATASETS_VALIDATED = "datasets_validated"
    GROUPING_JUDGEMENTS = "grouping_judgements"
    APPLYING_STRATEGY = "applying_strategy"
    SAMPLE_AGGREGATED = "sample_aggregated"
    DATASET_CREATED = "aggregated_dataset_created"
    AGGREGATE_SUMMARY_WRITTEN = "aggregation_summary_written"
    AGGREGATE_COMPLETE = "aggregation_complete"
    LOGS_SAVED = "logs_saved"
    WARNINGS_COLLECTED = "warnings_collected"


class AggregateWriteEvent(str, Enum):
    """Log events for aggregation write operations."""

    WRITE_AGGREGATION_STRATEGIES = "write_aggregation_strategies"
    WRITE_AGGREGATE_RUN_CONFIGS = "write_aggregate_run_configs"
    WRITE_AGGREGATE_RUNS = "write_aggregate_runs"
    WRITE_AGGREGATED_DATASETS = "write_aggregated_datasets"
    WRITE_DATASET_VOTES = "write_dataset_votes"
    WRITE_AGGREGATED_VOTES = "write_aggregated_votes"
    WRITE_AGGREGATION_VOTES = "write_aggregation_votes"
    WRITE_COMPLETE = "write_complete"
