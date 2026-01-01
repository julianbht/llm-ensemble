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
