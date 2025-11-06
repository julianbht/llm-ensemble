"""UUID computation helpers for deterministic entity identification.

All UUIDs are computed using UUIDv5 with entity-specific namespace UUIDs.
This ensures:
- Same logical entity → same UUID (idempotent writes)
- Different entity types → different UUIDs (no collisions)
- No need to query database before insert (just compute UUID)
"""

import uuid


# ========================================================================
# Namespace UUIDs for deterministic UUIDv5 generation
# ========================================================================
# Each entity type has its own namespace UUID to ensure no collisions
# between different entity types even if they have the same natural key.

NAMESPACE_DATASET = uuid.UUID('f0e1d2c3-b4a5-9687-7654-321fedcba098')
NAMESPACE_QUERY = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
NAMESPACE_DOCUMENT = uuid.UUID('b2c3d4e5-f678-90ab-cdef-123456789012')
NAMESPACE_JUDGING_SAMPLE = uuid.UUID('c3d4e5f6-7890-abcd-ef12-34567890abcd')
NAMESPACE_INGEST_RUN = uuid.UUID('d4e5f678-90ab-cdef-1234-567890abcdef')
NAMESPACE_INFER_RUN = uuid.UUID('e5f67890-abcd-ef12-3456-7890abcdef12')
NAMESPACE_AGGREGATE_RUN = uuid.UUID('f6789012-3456-7890-abcd-ef1234567890')
NAMESPACE_LLM_REQUEST = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456789')
NAMESPACE_LLM_RESPONSE = uuid.UUID('b1c2d3e4-f567-890a-bcde-f12345678901')
NAMESPACE_LLM_SCORE = uuid.UUID('c2d3e4f5-6789-0abc-def1-234567890abc')
NAMESPACE_LLM_JUDGEMENT = uuid.UUID('d3e4f567-890a-bcde-f123-4567890abcde')
NAMESPACE_INFER_WARNING = uuid.UUID('1a2b3c4d-5e6f-7890-abcd-ef1234567890')
NAMESPACE_AGGREGATED_SCORE = uuid.UUID('e4f56789-0abc-def1-2345-67890abcdef1')
NAMESPACE_AGGREGATED_JUDGEMENT = uuid.UUID('f5678901-abcd-ef12-3456-7890abcdef12')

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


def compute_query_uuid(dataset_id: uuid.UUID, external_id: str) -> uuid.UUID:
    """Compute deterministic UUID for a Query.

    Args:
        dataset_id: Dataset UUID (matches foreign key in database)
        external_id: Query's external identifier from the dataset

    Returns:
        Deterministic UUID for this query

    Example:
        >>> dataset_id = compute_dataset_uuid("msmarco")
        >>> compute_query_uuid(dataset_id, "q123")
        UUID('...')
    """
    natural_key = f"{dataset_id}:{external_id}"
    return uuid.uuid5(NAMESPACE_QUERY, natural_key)


def compute_document_uuid(dataset_id: uuid.UUID, external_id: str) -> uuid.UUID:
    """Compute deterministic UUID for a Document.

    Args:
        dataset_id: Dataset UUID (matches foreign key in database)
        external_id: Document's external identifier from the dataset

    Returns:
        Deterministic UUID for this document

    Example:
        >>> dataset_id = compute_dataset_uuid("msmarco")
        >>> compute_document_uuid(dataset_id, "d456")
        UUID('...')
    """
    natural_key = f"{dataset_id}:{external_id}"
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


# ========================================================================
# Infer Entity UUIDs
# ========================================================================

def compute_llm_judgement_uuid(
    judging_sample_id: uuid.UUID,
    infer_run_id: uuid.UUID
) -> uuid.UUID:
    """Compute deterministic UUID for an LLMJudgement.

    Args:
        judging_sample_id: UUID of the JudgingSample being judged
        infer_run_id: UUID of the InferRun

    Returns:
        Deterministic UUID for this LLM judgement

    Example:
        >>> sample_id = uuid.UUID('...')
        >>> run_id = compute_infer_run_uuid("20240101_130000_def456")
        >>> compute_llm_judgement_uuid(sample_id, run_id)
        UUID('...')
    """
    natural_key = f"{judging_sample_id}:{infer_run_id}"
    return uuid.uuid5(NAMESPACE_LLM_JUDGEMENT, natural_key)


def compute_infer_warning_uuid(
    judgement_id: uuid.UUID,
    stage: str,
    code: str,
    message: str
) -> uuid.UUID:
    """Compute deterministic UUID for an InferWarning.

    Args:
        judgement_id: UUID of the LLMJudgement this warning belongs to
        stage: Warning stage (PROMPT/PROVIDER/PARSER)
        code: Warning code
        message: Warning message

    Returns:
        Deterministic UUID for this warning

    Example:
        >>> judgement_id = uuid.UUID('...')
        >>> compute_infer_warning_uuid(
        ...     judgement_id,
        ...     "PARSER",
        ...     "FIELD_ERROR",
        ...     "Missing confidence field"
        ... )
        UUID('...')
    """
    # Hash message to keep natural key reasonable length
    import hashlib
    message_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
    natural_key = f"{judgement_id}:{stage}:{code}:{message_hash}"
    return uuid.uuid5(NAMESPACE_INFER_WARNING, natural_key)