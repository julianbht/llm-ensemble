"""UUID computation helpers for deterministic entity identification.

All UUIDs are computed using UUIDv5 with entity-specific namespace UUIDs.
This ensures:
- Same logical entity → same UUID (idempotent writes)
- Different entity types → different UUIDs (no collisions)
- No need to query database before insert (just compute UUID)
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
NAMESPACE_INGEST_RUN = uuid.UUID('d4e5f678-90ab-cdef-1234-567890abcdef')
NAMESPACE_INFER_RUN = uuid.UUID('e5f67890-abcd-ef12-3456-7890abcdef12')
NAMESPACE_AGGREGATE_RUN = uuid.UUID('f6789012-3456-7890-abcd-ef1234567890')
NAMESPACE_LLM_REQUEST = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456789')
NAMESPACE_LLM_RESPONSE = uuid.UUID('b1c2d3e4-f567-890a-bcde-f12345678901')
NAMESPACE_LLM_SCORE = uuid.UUID('c2d3e4f5-6789-0abc-def1-234567890abc')
NAMESPACE_LLM_JUDGEMENT = uuid.UUID('d3e4f567-890a-bcde-f123-4567890abcde')
NAMESPACE_INFER_WARNING = uuid.UUID('1a2b3c4d-5e6f-7890-abcd-ef1234567890')
NAMESPACE_PROMPT_TEMPLATE = uuid.UUID('2b3c4d5e-6f78-9012-3456-789abcdef012')
NAMESPACE_PROMPT_CONFIG = uuid.UUID('3c4d5e6f-7890-1234-5678-90abcdef1234')
NAMESPACE_MODEL_CONFIG = uuid.UUID('4d5e6f78-9012-3456-7890-1abcdef12345')
NAMESPACE_PROVIDER = uuid.UUID('5e6f7890-1234-5678-9012-3abcdef12346')
NAMESPACE_AGGREGATED_SCORE = uuid.UUID('e4f56789-0abc-def1-2345-67890abcdef1')
NAMESPACE_AGGREGATED_JUDGEMENT = uuid.UUID('f5678901-abcd-ef12-3456-7890abcdef12')

# ========================================================================
# Run Info UUIDs
# ========================================================================

def compute_ingest_run_uuid(run_id: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_INGEST_RUN, run_id)


def compute_infer_run_uuid(run_id: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_INFER_RUN, run_id)


def compute_aggregate_run_uuid(run_id: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_AGGREGATE_RUN, run_id)


# ========================================================================
# Core Entity UUIDs (from ingest)
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
    doc_id: str
) -> uuid.UUID:
    natural_key = f"{query_id}:{doc_id}"
    return uuid.uuid5(NAMESPACE_JUDGING_SAMPLE, natural_key)

def compute_ingest_run_uuid(run_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_INGEST_RUN, run_name)


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


# ========================================================================
# Config Entity UUIDs
# ========================================================================

def compute_prompt_template_uuid(template_text: str) -> uuid.UUID:
    # Use SHA-256 hash of template text for content-addressable UUID
    template_hash = hashlib.sha256(template_text.encode('utf-8')).hexdigest()
    return uuid.uuid5(NAMESPACE_PROMPT_TEMPLATE, template_hash)


def compute_prompt_config_uuid(config_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_PROMPT_CONFIG, config_name)


def compute_model_config_uuid(config_name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_MODEL_CONFIG, config_name)


def compute_provider_uuid(provider_name: str) -> uuid.UUID:
    """Compute deterministic UUID for a Provider.

    Args:
        provider_name: Provider name (e.g., "openrouter", "ollama", "hf")

    Returns:
        Deterministic UUID for this provider

    Example:
        >>> compute_provider_uuid("openrouter")
        UUID('...')
    """
    return uuid.uuid5(NAMESPACE_PROVIDER, provider_name)