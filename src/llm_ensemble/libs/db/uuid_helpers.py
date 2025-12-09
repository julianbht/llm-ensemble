"""UUID computation helpers for deterministic entity identification.

All UUIDs are computed using UUIDv5 with entity-specific namespace UUIDs.
This ensures:
- Same logical entity → same UUID (idempotent writes)
- Different entity types → different UUIDs (no collisions)

Note: Ingest entities (Query, Document, JudgingSample, etc.), Infer entities
(Provider, ModelConfig, LLMJudgement, InferRunInfo, etc.), and Aggregate entities
(AggregationStrategy, AggregatedVote, AggregatedDataset, AggregateRunInfo, etc.)
now use random UUIDs with constraint-based duplicate detection, so their UUID
computation functions have been removed.
"""

import uuid
import hashlib

# ========================================================================
# Namespace UUIDs for deterministic UUIDv5 generation
# ========================================================================
# Each entity type has its own namespace UUID to ensure no collisions
# between different entity types even if they have the same natural key.

NAMESPACE_JUDGED_DATASET = uuid.UUID('c5d6e7f8-9012-cdef-3456-67890abcdef0')
NAMESPACE_LLM_PROMPT = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456789')
NAMESPACE_LLM_PROMPT_TEXT = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456790')
NAMESPACE_LLM_RESPONSE_TEXT = uuid.UUID('b1c2d3e4-f567-8901-2345-6789abcdef01')
NAMESPACE_LLM_INVOCATION_METRICS = uuid.UUID('b2c3d4e5-f678-90ab-cdef-123456789abc')
NAMESPACE_LLM_SCORE = uuid.UUID('c2d3e4f5-6789-0abc-def1-234567890abc')
NAMESPACE_LLM_JUDGEMENT = uuid.UUID('d3e4f567-890a-bcde-f123-4567890abcde')


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
