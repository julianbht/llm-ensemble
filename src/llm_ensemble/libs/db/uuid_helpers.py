"""UUID computation helpers for deterministic entity identification.

All UUIDs are computed using UUIDv5 with entity-specific namespace UUIDs.
This ensures:
- Same logical entity → same UUID (idempotent writes)
- Different entity types → different UUIDs (no collisions)
- No need to query database before insert (just compute UUID)
"""

import uuid
from llm_ensemble.libs.db.base import (
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

# ========================================================================
# Run Info UUIDs
# ========================================================================

def compute_ingest_run_uuid(run_id: str) -> uuid.UUID:
    """Compute deterministic UUID for an IngestRunInfo.
    
    Args:
        run_id: Run identifier (timestamp-based)
    
    Returns:
        Deterministic UUID for this ingest run
    
    Example:
        >>> compute_ingest_run_uuid("20240101_120000_abc123")
        UUID('...')
    """
    return uuid.uuid5(NAMESPACE_INGEST_RUN, run_id)


def compute_infer_run_uuid(run_id: str) -> uuid.UUID:
    """Compute deterministic UUID for an InferRunInfo.
    
    Args:
        run_id: Run identifier (timestamp-based)
    
    Returns:
        Deterministic UUID for this infer run
    
    Example:
        >>> compute_infer_run_uuid("20240101_130000_def456")
        UUID('...')
    """
    return uuid.uuid5(NAMESPACE_INFER_RUN, run_id)


def compute_aggregate_run_uuid(run_id: str) -> uuid.UUID:
    """Compute deterministic UUID for an AggregateRunInfo.
    
    Args:
        run_id: Run identifier (timestamp-based)
    
    Returns:
        Deterministic UUID for this aggregate run
    
    Example:
        >>> compute_aggregate_run_uuid("20240101_140000_ghi789")
        UUID('...')
    """
    return uuid.uuid5(NAMESPACE_AGGREGATE_RUN, run_id)


# ========================================================================
# Core Entity UUIDs (from ingest)
# ========================================================================

def compute_dataset_uuid(name: str) -> uuid.UUID:
    """Compute deterministic UUID for a Dataset.
    
    Args:
        name: Dataset name (e.g., 'msmarco', 'trec-covid', 'llmjudge')
    
    Returns:
        Deterministic UUID for this dataset
    
    Example:
        >>> compute_dataset_uuid("msmarco")
        UUID('...')
    """
    return uuid.uuid5(NAMESPACE_DATASET, name)


def compute_query_uuid(dataset: str, external_id: str) -> uuid.UUID:
    """Compute deterministic UUID for a Query.
    
    Args:
        dataset: Dataset name (e.g., 'msmarco', 'trec-covid')
        external_id: Query's external identifier from the dataset
    
    Returns:
        Deterministic UUID for this query
    
    Example:
        >>> compute_query_uuid("msmarco", "q123")
        UUID('...')
    """
    natural_key = f"{dataset}:{external_id}"
    return uuid.uuid5(NAMESPACE_QUERY, natural_key)


def compute_document_uuid(dataset: str, external_id: str) -> uuid.UUID:
    """Compute deterministic UUID for a Document.
    
    Args:
        dataset: Dataset name (e.g., 'msmarco', 'trec-covid')
        external_id: Document's external identifier from the dataset
    
    Returns:
        Deterministic UUID for this document
    
    Example:
        >>> compute_document_uuid("msmarco", "d456")
        UUID('...')
    """
    natural_key = f"{dataset}:{external_id}"
    return uuid.uuid5(NAMESPACE_DOCUMENT, natural_key)


def compute_judging_sample_uuid(
    query_id: str,
    doc_id: str
) -> uuid.UUID:
    """Compute deterministic UUID for a JudgingSample.
    
    Args:
        dataset: Dataset name
        query_external_id: Query's external identifier
        doc_external_id: Document's external identifier
    
    Returns:
        Deterministic UUID for this judging sample
    
    Example:
        >>> compute_judging_sample_uuid("msmarco", "q123", "d456")
        UUID('...')
    """
    natural_key = f"{query_id}:{doc_id}"
    return uuid.uuid5(NAMESPACE_JUDGING_SAMPLE, natural_key)

def compute_ingest_run_uuid(run_name: str) -> uuid.UUID:
    """Compute deterministic UUID for an IngestRunInfo.
    
    Args:
        run_name: Run identifier (timestamp-based)
    
    Returns:
        Deterministic UUID for this ingest run
    
    Example:
        >>> compute_ingest_run_uuid("20240101_120000_abc123")
        UUID('...')
    """
    return uuid.uuid5(NAMESPACE_INGEST_RUN, run_name)