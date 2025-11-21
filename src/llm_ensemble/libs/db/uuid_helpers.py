"""UUID computation helpers for deterministic entity identification.

All UUIDs are computed using UUIDv5 with entity-specific namespace UUIDs.
This ensures:
- Same logical entity → same UUID (idempotent writes)
- Different entity types → different UUIDs (no collisions)
- Because using combined UUID's as primary keys is essentially the same as enforcing
unique constraints on the db level, we ensure that the uuid computation matches the 
db unique constraints with automated tests.
"""

import uuid
import hashlib

# ========================================================================
# Namespace UUIDs for deterministic UUIDv5 generation
# ========================================================================
# Each entity type has its own namespace UUID to ensure no collisions
# between different entity types even if they have the same natural key.

NAMESPACE_DATASET = uuid.UUID('f0e1d2c3-b4a5-9687-7654-321fedcba098')
NAMESPACE_QUERY = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
NAMESPACE_DOCUMENT = uuid.UUID('b2c3d4e5-f678-90ab-cdef-123456789012')
NAMESPACE_JUDGING_SAMPLE = uuid.UUID('c3d4e5f6-7890-abcd-ef12-34567890abcd')
NAMESPACE_NORMALIZED_DATASET = uuid.UUID('c4d5e6f7-8901-bcde-f234-567890abcdef')
NAMESPACE_JUDGED_DATASET = uuid.UUID('c5d6e7f8-9012-cdef-3456-67890abcdef0')
NAMESPACE_INGEST_RUN = uuid.UUID('d4e5f678-90ab-cdef-1234-567890abcdef')
NAMESPACE_INFER_RUN = uuid.UUID('e5f67890-abcd-ef12-3456-7890abcdef12')
NAMESPACE_AGGREGATE_RUN = uuid.UUID('f6789012-3456-7890-abcd-ef1234567890')
NAMESPACE_LLM_REQUEST = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456789')
NAMESPACE_LLM_SCORE = uuid.UUID('c2d3e4f5-6789-0abc-def1-234567890abc')
NAMESPACE_LLM_JUDGEMENT = uuid.UUID('d3e4f567-890a-bcde-f123-4567890abcde')
NAMESPACE_INFER_WARNING = uuid.UUID('1a2b3c4d-5e6f-7890-abcd-ef1234567890')
NAMESPACE_PROMPT_TEMPLATE = uuid.UUID('2b3c4d5e-6f78-9012-3456-789abcdef012')
NAMESPACE_PROMPT_CONFIG = uuid.UUID('3c4d5e6f-7890-1234-5678-90abcdef1234')
NAMESPACE_MODEL_SPEC = uuid.UUID('4d5e6f78-9012-3456-7890-1abcdef12345')
NAMESPACE_PROVIDER = uuid.UUID('5e6f7890-1234-5678-9012-3abcdef12346')
NAMESPACE_PARSER_SPEC = uuid.UUID('6f789012-3456-7890-abcd-1234def56789')
NAMESPACE_LLM_CALL = uuid.UUID('789012ab-cdef-1234-5678-90abcdef1234')

# Aggregate namespace UUIDs
NAMESPACE_AGGREGATION_SPEC = uuid.UUID('890123bc-def1-2345-6789-0abcdef12345')
NAMESPACE_AGGREGATED_SCORE = uuid.UUID('a12345de-f123-4567-890a-bcdef1234567')
NAMESPACE_AGGREGATED_SCORE_LLM_CALL = uuid.UUID('b23456ef-1234-5678-9012-3456789abcde')

# ========================================================================
# Run Info UUIDs
# ========================================================================

def compute_ingest_run_uuid(run_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_INGEST_RUN, run_name)

def compute_infer_run_uuid(run_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_INFER_RUN, run_name)

def compute_aggregate_run_uuid(run_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_AGGREGATE_RUN, run_name)


# ========================================================================
# Ingest Entitie UUIDs
# ========================================================================

def compute_dataset_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_DATASET, name)


def compute_query_uuid(dataset_id: uuid.UUID, external_id: str) -> uuid.UUID:
    natural_key = f"{dataset_id}:{external_id}"
    return uuid.uuid5(NAMESPACE_QUERY, natural_key)


def compute_document_uuid(dataset_id: uuid.UUID, external_id: str) -> uuid.UUID:
    natural_key = f"{dataset_id}:{external_id}"
    return uuid.uuid5(NAMESPACE_DOCUMENT, natural_key)


def compute_judging_sample_uuid(
    query_id: str,
    document_id: str
) -> uuid.UUID:
    natural_key = f"{query_id}:{document_id}"
    return uuid.uuid5(NAMESPACE_JUDGING_SAMPLE, natural_key)


def compute_normalized_dataset_fingerprint(samples: list) -> str:
    """Compute deterministic SHA256 fingerprint from sorted sample IDs.

    The fingerprint uniquely identifies a specific set of judging samples,
    enabling idempotent ingest runs and efficient validation in aggregate CLI.

    Args:
        samples: List of JudgingSample objects (must be sorted by ID)

    Returns:
        SHA256 hash (64 character hex string) of sorted sample IDs
    """
    # Extract and sort sample IDs (already sorted, but ensure determinism)
    sorted_ids = sorted([str(s.id) for s in samples])

    # Create comma-separated string of UUIDs
    id_string = ",".join(sorted_ids)

    # Compute SHA256 hash
    return hashlib.sha256(id_string.encode()).hexdigest()


def compute_normalized_dataset_uuid(fingerprint: str) -> uuid.UUID:
    """Compute deterministic UUID for NormalizedDataset from fingerprint.

    Args:
        fingerprint: SHA256 hash of sorted sample IDs

    Returns:
        Deterministic UUID based on fingerprint
    """
    return uuid.uuid5(NAMESPACE_NORMALIZED_DATASET, fingerprint)


def compute_judged_dataset_fingerprint(llm_calls: list) -> str:
    """Compute deterministic SHA256 fingerprint from sorted LLMCall IDs.

    The fingerprint uniquely identifies a specific set of LLM judgements,
    enabling idempotent infer runs and efficient validation in aggregate CLI.

    Args:
        llm_calls: List of LLMCall objects (domain or ORM) with .id attribute

    Returns:
        SHA256 hash (64 character hex string) of sorted call IDs
    """
    sorted_ids = sorted([str(c.id) for c in llm_calls])
    id_string = ",".join(sorted_ids)
    return hashlib.sha256(id_string.encode()).hexdigest()


def compute_judged_dataset_uuid(fingerprint: str) -> uuid.UUID:
    """Compute deterministic UUID for JudgedDataset from fingerprint.

    Args:
        fingerprint: SHA256 hash of sorted LLMCall IDs

    Returns:
        Deterministic UUID based on fingerprint
    """
    return uuid.uuid5(NAMESPACE_JUDGED_DATASET, fingerprint)


# ========================================================================
# Infer Entity UUIDs
# ========================================================================

def compute_llm_judgement_uuid(
    judging_sample_id: uuid.UUID,
    infer_run_id: uuid.UUID
) -> uuid.UUID:
    natural_key = f"{judging_sample_id}:{infer_run_id}"
    return uuid.uuid5(NAMESPACE_LLM_JUDGEMENT, natural_key)


def compute_infer_warning_uuid(
    judgement_id: uuid.UUID,
    stage: str,
    code: str,
    message: str
) -> uuid.UUID:
    # Hash message to keep natural key reasonable length
    message_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
    natural_key = f"{judgement_id}:{stage}:{code}:{message_hash}"
    return uuid.uuid5(NAMESPACE_INFER_WARNING, natural_key)

def compute_prompt_template_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_PROMPT_TEMPLATE, name)


def compute_prompt_config_uuid(config_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_PROMPT_CONFIG, config_name)


def compute_model_spec_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_MODEL_SPEC, name)


def compute_provider_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_PROVIDER, name)


def compute_parser_spec_uuid(parser_module: str, parser_class: str, code_hash: str) -> uuid.UUID:
    natural_key = f"{parser_module}:{parser_class}:{code_hash}"
    return uuid.uuid5(NAMESPACE_PARSER_SPEC, natural_key)


def compute_llm_request_uuid(prompt: str, judging_sample_id: uuid.UUID) -> uuid.UUID:
    # Hash prompt to keep natural key reasonable length
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    natural_key = f"{judging_sample_id}:{prompt_hash}"
    return uuid.uuid5(NAMESPACE_LLM_REQUEST, natural_key)


def compute_llm_score_uuid(parser_spec_id: uuid.UUID, raw_response: str) -> uuid.UUID:
    # Hash raw response to keep natural key reasonable length
    response_hash = hashlib.sha256(raw_response.encode()).hexdigest()
    natural_key = f"{parser_spec_id}:{response_hash}"
    return uuid.uuid5(NAMESPACE_LLM_SCORE, natural_key)


def compute_llm_call_uuid(llm_request_id: uuid.UUID, infer_run_id: uuid.UUID) -> uuid.UUID:
    natural_key = f"{llm_request_id}:{infer_run_id}"
    return uuid.uuid5(NAMESPACE_LLM_CALL, natural_key)

# ========================================================================
# Aggregate Entity UUIDs
# ========================================================================

def compute_aggregation_spec_uuid(name: str) -> uuid.UUID:
    """Compute deterministic UUID for aggregation spec.

    Args:
        name: Spec name (e.g., 'majority_vote', 'weighted_majority')

    Returns:
        Deterministic UUID based on spec name
    """
    return uuid.uuid5(NAMESPACE_AGGREGATION_SPEC, name)


def compute_aggregated_score_uuid(
    aggregate_run_id: uuid.UUID,
    llm_call_ids: tuple[uuid.UUID, ...]
) -> uuid.UUID:
    """Compute deterministic UUID for aggregated score.

    The score is uniquely identified by the aggregate run and the set of LLM calls
    being aggregated (order-independent set, sorted for determinism).

    Args:
        aggregate_run_id: UUID of the aggregate run
        llm_call_ids: Tuple of LLM call UUIDs being aggregated (will be sorted)

    Returns:
        Deterministic UUID based on composite key
    """
    # Sort for deterministic ordering (set of calls, not ordered list)
    sorted_call_ids = ":".join(str(cid) for cid in sorted(llm_call_ids))
    natural_key = f"{aggregate_run_id}:{sorted_call_ids}"
    return uuid.uuid5(NAMESPACE_AGGREGATED_SCORE, natural_key)


def compute_aggregated_score_llm_call_uuid(
    aggregated_score_id: uuid.UUID,
    llm_call_id: uuid.UUID
) -> uuid.UUID:
    """Compute deterministic UUID for aggregated score LLM call join table.

    Join table linking aggregated scores to their constituent LLM calls.

    Args:
        aggregated_score_id: UUID of the aggregated score
        llm_call_id: UUID of the LLM call (individual model judgement)

    Returns:
        Deterministic UUID based on composite key
    """
    natural_key = f"{aggregated_score_id}:{llm_call_id}"
    return uuid.uuid5(NAMESPACE_AGGREGATED_SCORE_LLM_CALL, natural_key)
