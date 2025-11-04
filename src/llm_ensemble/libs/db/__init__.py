"""Database utilities for LLM Ensemble.

Provides SQLAlchemy ORM base, engine factory, session management,
and UUID computation helpers for deterministic entity identification.
"""

from llm_ensemble.libs.db.base import (
    Base,
    get_engine,
    create_all_tables,
    NAMESPACE_DATASET,
    NAMESPACE_QUERY,
    NAMESPACE_DOCUMENT,
    NAMESPACE_JUDGING_SAMPLE,
    NAMESPACE_INGEST_RUN,
    NAMESPACE_INFER_RUN,
    NAMESPACE_AGGREGATE_RUN,
    NAMESPACE_LLM_REQUEST,
    NAMESPACE_LLM_RESPONSE,
    NAMESPACE_LLM_SCORE,
    NAMESPACE_LLM_JUDGEMENT,
    NAMESPACE_AGGREGATED_SCORE,
    NAMESPACE_AGGREGATED_JUDGEMENT,
)

from llm_ensemble.libs.db.session import (
    get_session,
    session_context,
)

from llm_ensemble.libs.db.uuid_helpers import (
    compute_dataset_uuid,
    compute_query_uuid,
    compute_document_uuid,
    compute_judging_sample_uuid,
    compute_ingest_run_uuid
)

__all__ = [
    # Base and engine
    "Base",
    "get_engine",
    "create_all_tables",
    # Session management
    "get_session",
    "session_context",
    # Namespace UUIDs
    "NAMESPACE_DATASET",
    "NAMESPACE_QUERY",
    "NAMESPACE_DOCUMENT",
    "NAMESPACE_JUDGING_SAMPLE",
    "NAMESPACE_INGEST_RUN",
    "NAMESPACE_INFER_RUN",
    "NAMESPACE_AGGREGATE_RUN",
    "NAMESPACE_LLM_REQUEST",
    "NAMESPACE_LLM_RESPONSE",
    "NAMESPACE_LLM_SCORE",
    "NAMESPACE_LLM_JUDGEMENT",
    "NAMESPACE_AGGREGATED_SCORE",
    "NAMESPACE_AGGREGATED_JUDGEMENT",
    # UUID helpers
    "compute_dataset_uuid",
    "compute_query_uuid",
    "compute_document_uuid",
    "compute_judging_sample_uuid",
    "compute_ingest_run_uuid",
    "compute_infer_run_uuid",
    "compute_aggregate_run_uuid",
    "compute_llm_request_uuid",
    "compute_llm_response_uuid",
    "compute_llm_score_uuid",
    "compute_llm_judgement_uuid",
    "compute_aggregated_score_uuid",
    "compute_aggregated_judgement_uuid",
]
