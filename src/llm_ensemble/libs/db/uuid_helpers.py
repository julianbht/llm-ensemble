"""UUID computation helpers for deterministic entity identification.

All UUIDs are computed using UUIDv5 with entity-specific namespace UUIDs.
This ensures:
- Same logical entity → same UUID (idempotent writes)
- Different entity types → different UUIDs (no collisions)

Note: Ingest entities (Query, Document, JudgingSample, etc.) now use random UUIDs
with constraint-based duplicate detection, so their UUID computation functions have
been removed.
"""

import uuid
import hashlib

# ========================================================================
# Namespace UUIDs for deterministic UUIDv5 generation
# ========================================================================
# Each entity type has its own namespace UUID to ensure no collisions
# between different entity types even if they have the same natural key.

NAMESPACE_JUDGED_DATASET = uuid.UUID('c5d6e7f8-9012-cdef-3456-67890abcdef0')
NAMESPACE_INFER_RUN = uuid.UUID('e5f67890-abcd-ef12-3456-7890abcdef12')
NAMESPACE_AGGREGATE_RUN = uuid.UUID('f6789012-3456-7890-abcd-ef1234567890')
NAMESPACE_LLM_PROMPT = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456789')
NAMESPACE_LLM_PROMPT_TEXT = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456790')
NAMESPACE_LLM_RESPONSE_TEXT = uuid.UUID('b1c2d3e4-f567-8901-2345-6789abcdef01')
NAMESPACE_LLM_INVOCATION_METRICS = uuid.UUID('b2c3d4e5-f678-90ab-cdef-123456789abc')
NAMESPACE_LLM_SCORE = uuid.UUID('c2d3e4f5-6789-0abc-def1-234567890abc')
NAMESPACE_LLM_JUDGEMENT = uuid.UUID('d3e4f567-890a-bcde-f123-4567890abcde')
NAMESPACE_INFER_WARNING = uuid.UUID('1a2b3c4d-5e6f-7890-abcd-ef1234567890')
NAMESPACE_PROMPT_TEMPLATE = uuid.UUID('2b3c4d5e-6f78-9012-3456-789abcdef012')
NAMESPACE_PROMPT_CONFIG = uuid.UUID('3c4d5e6f-7890-1234-5678-90abcdef1234')
NAMESPACE_MODEL = uuid.UUID('3d4e5f67-8901-2345-6789-0abcdef12344')
NAMESPACE_MODEL_CONFIG = uuid.UUID('4d5e6f78-9012-3456-7890-1abcdef12345')
NAMESPACE_PROVIDER = uuid.UUID('5e6f7890-1234-5678-9012-3abcdef12346')
NAMESPACE_PARSER_SPEC = uuid.UUID('6f789012-3456-7890-abcd-1234def56789')
NAMESPACE_DATASET_JUDGEMENT = uuid.UUID('7890abcd-ef12-3456-7890-abcdef123456')
NAMESPACE_AGGREGATION_SPEC = uuid.UUID('890123bc-def1-2345-6789-0abcdef12345')
NAMESPACE_AGGREGATED_DATASET = uuid.UUID('901234cd-ef12-3456-789a-bcdef1234567')
NAMESPACE_DATASET_VOTE = uuid.UUID('a12345de-f123-4567-890a-bcdef1234567')
NAMESPACE_AGGREGATED_VOTE = uuid.UUID('b23456ef-1234-5678-901a-bcdef1234568')
NAMESPACE_AGGREGATION_VOTE = uuid.UUID('c34567f0-1234-5678-9012-3456789abcde')

# ========================================================================
# Run Info UUIDs
# ========================================================================

def compute_infer_run_uuid(run_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_INFER_RUN, run_name)

def compute_aggregate_run_uuid(run_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_AGGREGATE_RUN, run_name)


def compute_judged_dataset_fingerprint(dataset_sample_ids: list[uuid.UUID]) -> str:
    """Compute fingerprint from sorted dataset_sample IDs.

    Args:
        dataset_sample_ids: List of dataset_sample UUIDs

    Returns:
        SHA256 hash of sorted, comma-separated UUID strings
    """
    sorted_ids = sorted([str(id) for id in dataset_sample_ids])
    id_string = ",".join(sorted_ids)
    return hashlib.sha256(id_string.encode()).hexdigest()


def compute_judged_dataset_uuid(fingerprint: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_JUDGED_DATASET, fingerprint)

# ========================================================================
# Aggregate Entity UUIDs
# ========================================================================

def compute_aggregation_spec_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_AGGREGATION_SPEC, name)


def compute_aggregated_dataset_fingerprint(dataset_sample_ids: list[uuid.UUID]) -> str:
    """Compute fingerprint from sorted dataset_sample IDs.

    Args:
        dataset_sample_ids: List of dataset_sample UUIDs

    Returns:
        SHA256 hash of sorted, comma-separated UUID strings
    """
    sorted_ids = sorted([str(id) for id in dataset_sample_ids])
    id_string = ",".join(sorted_ids)
    return hashlib.sha256(id_string.encode()).hexdigest()


def compute_aggregated_dataset_uuid(fingerprint: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_AGGREGATED_DATASET, fingerprint)


def compute_dataset_vote_uuid(
    aggregated_dataset_id: uuid.UUID,
    sequence_number: int
) -> uuid.UUID:
    natural_key = f"{aggregated_dataset_id}:{sequence_number}"
    return uuid.uuid5(NAMESPACE_DATASET_VOTE, natural_key)


def compute_aggregated_vote_uuid(
    dataset_sample_id: uuid.UUID,
    aggregation_spec_id: uuid.UUID
) -> uuid.UUID:
    natural_key = f"{dataset_sample_id}:{aggregation_spec_id}"
    return uuid.uuid5(NAMESPACE_AGGREGATED_VOTE, natural_key)


def compute_aggregation_vote_uuid(
    aggregated_vote_id: uuid.UUID,
    llm_judgement_id: uuid.UUID
) -> uuid.UUID:
    natural_key = f"{aggregated_vote_id}:{llm_judgement_id}"
    return uuid.uuid5(NAMESPACE_AGGREGATION_VOTE, natural_key)
