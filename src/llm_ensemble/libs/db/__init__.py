"""Database utilities for LLM Ensemble.

Provides SQLAlchemy ORM base, engine factory, session management,
and UUID computation helpers for deterministic entity identification.
"""

from llm_ensemble.libs.db.base import (
    Base,
    get_engine,
    create_schemas,
    create_all_tables,
)

from llm_ensemble.libs.db.session import (
    get_session,
    session_context,
)

from llm_ensemble.libs.db.uuid_helpers import (
    compute_query_uuid,
    compute_document_uuid,
    compute_judging_sample_uuid,
    compute_normalized_dataset_fingerprint,
    compute_normalized_dataset_uuid,
    compute_dataset_sample_uuid,
    compute_judged_dataset_fingerprint,
    compute_judged_dataset_uuid,
    compute_ingest_run_uuid,
    compute_infer_run_uuid,
    compute_aggregate_run_uuid,
    compute_aggregation_spec_uuid,
    compute_aggregated_dataset_fingerprint,
    compute_aggregated_dataset_uuid,
    compute_dataset_vote_uuid,
    compute_aggregated_vote_uuid,
    compute_aggregation_vote_uuid,
)

from llm_ensemble.libs.db.utcnow import utcnow

__all__ = [
    # Base and engine
    "Base",
    "get_engine",
    "create_schemas",
    "create_all_tables",
    # Session management
    "get_session",
    "session_context",
    # UUID helpers - ingest
    "compute_query_uuid",
    "compute_document_uuid",
    "compute_judging_sample_uuid",
    "compute_normalized_dataset_fingerprint",
    "compute_normalized_dataset_uuid",
    "compute_dataset_sample_uuid",
    "compute_ingest_run_uuid",
    # UUID helpers - infer
    "compute_judged_dataset_fingerprint",
    "compute_judged_dataset_uuid",
    "compute_infer_run_uuid",
    # UUID helpers - aggregate
    "compute_aggregate_run_uuid",
    "compute_aggregation_spec_uuid",
    "compute_aggregated_dataset_fingerprint",
    "compute_aggregated_dataset_uuid",
    "compute_dataset_vote_uuid",
    "compute_aggregated_vote_uuid",
    "compute_aggregation_vote_uuid",
    # Other db helpers
    "utcnow",
]
