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

NAMESPACE_QUERY = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
NAMESPACE_DOCUMENT = uuid.UUID('b2c3d4e5-f678-90ab-cdef-123456789012')
NAMESPACE_JUDGING_SAMPLE = uuid.UUID('c3d4e5f6-7890-abcd-ef12-34567890abcd')
NAMESPACE_NORMALIZED_DATASET = uuid.UUID('c4d5e6f7-8901-bcde-f234-567890abcdef')
NAMESPACE_DATASET_SAMPLE = uuid.UUID('c5d6e7f8-9012-cdef-1234-567890abcdef')
NAMESPACE_JUDGED_DATASET = uuid.UUID('c5d6e7f8-9012-cdef-3456-67890abcdef0')
NAMESPACE_INGEST_RUN = uuid.UUID('d4e5f678-90ab-cdef-1234-567890abcdef')
NAMESPACE_INFER_RUN = uuid.UUID('e5f67890-abcd-ef12-3456-7890abcdef12')
NAMESPACE_AGGREGATE_RUN = uuid.UUID('f6789012-3456-7890-abcd-ef1234567890')
NAMESPACE_LLM_PROMPT = uuid.UUID('a0b1c2d3-e4f5-6789-0abc-def123456789')
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

def compute_query_uuid(query_text: str) -> uuid.UUID:
    text_hash = hashlib.sha256(query_text.encode()).hexdigest()
    return uuid.uuid5(NAMESPACE_QUERY, text_hash)


def compute_document_uuid(doc_text: str) -> uuid.UUID:
    text_hash = hashlib.sha256(doc_text.encode()).hexdigest()
    return uuid.uuid5(NAMESPACE_DOCUMENT, text_hash)


def compute_judging_sample_uuid(
    query_id: str,
    document_id: str
) -> uuid.UUID:
    natural_key = f"{query_id}:{document_id}"
    return uuid.uuid5(NAMESPACE_JUDGING_SAMPLE, natural_key)


def compute_normalized_dataset_fingerprint(samples: list) -> str:
    # Extract and sort sample IDs (already sorted, but ensure determinism)
    sorted_ids = sorted([str(s.id) for s in samples])

    # Create comma-separated string of UUIDs
    id_string = ",".join(sorted_ids)

    # Compute SHA256 hash
    return hashlib.sha256(id_string.encode()).hexdigest()


def compute_normalized_dataset_uuid(fingerprint: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_NORMALIZED_DATASET, fingerprint)


def compute_dataset_sample_uuid(
    normalized_dataset_id: uuid.UUID,
    judging_sample_id: uuid.UUID
) -> uuid.UUID:
    natural_key = f"{normalized_dataset_id}:{judging_sample_id}"
    return uuid.uuid5(NAMESPACE_DATASET_SAMPLE, natural_key)


def compute_judged_dataset_fingerprint(judgements: list) -> str:
    sorted_ids = sorted([str(j.id) for j in judgements])
    id_string = ",".join(sorted_ids)
    return hashlib.sha256(id_string.encode()).hexdigest()


def compute_judged_dataset_uuid(fingerprint: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_JUDGED_DATASET, fingerprint)


# ========================================================================
# Infer Entity UUIDs
# ========================================================================

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


def compute_model_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_MODEL, name)


def compute_model_config_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_MODEL_CONFIG, name)


def compute_model_spec_uuid(name: str) -> uuid.UUID:
    """Deprecated: Use compute_model_config_uuid instead."""
    return uuid.uuid5(NAMESPACE_MODEL_CONFIG, name)


def compute_provider_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_PROVIDER, name)


def compute_parser_spec_uuid(parser_module: str, parser_class: str, code_hash: str) -> uuid.UUID:
    natural_key = f"{parser_module}:{parser_class}:{code_hash}"
    return uuid.uuid5(NAMESPACE_PARSER_SPEC, natural_key)


def compute_llm_prompt_uuid(prompt: str, judging_sample_id: uuid.UUID) -> uuid.UUID:
    # Hash prompt to keep natural key reasonable length
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    natural_key = f"{judging_sample_id}:{prompt_hash}"
    return uuid.uuid5(NAMESPACE_LLM_PROMPT, natural_key)


def compute_llm_response_text_uuid(raw_response: str) -> uuid.UUID:
    response_hash = hashlib.sha256(raw_response.encode()).hexdigest()
    return uuid.uuid5(NAMESPACE_LLM_RESPONSE_TEXT, response_hash)


def compute_llm_invocation_metrics_uuid(
    latency_ms: float,
    retries: int,
    cost_estimate_usd: float | None,
    generation_id: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None
) -> uuid.UUID:
    # Build natural key from all fields (handling None values)
    fields = [
        str(latency_ms),
        str(retries),
        str(cost_estimate_usd) if cost_estimate_usd is not None else "NULL",
        generation_id if generation_id else "NULL",
        str(prompt_tokens) if prompt_tokens is not None else "NULL",
        str(completion_tokens) if completion_tokens is not None else "NULL",
        str(total_tokens) if total_tokens is not None else "NULL"
    ]
    natural_key = ":".join(fields)
    return uuid.uuid5(NAMESPACE_LLM_INVOCATION_METRICS, natural_key)


def compute_llm_score_uuid(parser_spec_id: uuid.UUID, llm_response_text_id: uuid.UUID) -> uuid.UUID:
    natural_key = f"{parser_spec_id}:{llm_response_text_id}"
    return uuid.uuid5(NAMESPACE_LLM_SCORE, natural_key)


def compute_llm_judgement_uuid(
    llm_prompt_id: uuid.UUID,
    llm_response_text_id: uuid.UUID,
    llm_invocation_metrics_id: uuid.UUID
) -> uuid.UUID:
    natural_key = f"{llm_prompt_id}:{llm_response_text_id}:{llm_invocation_metrics_id}"
    return uuid.uuid5(NAMESPACE_LLM_JUDGEMENT, natural_key)

# ========================================================================
# Aggregate Entity UUIDs
# ========================================================================

def compute_aggregation_spec_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE_AGGREGATION_SPEC, name)


def compute_aggregated_score_uuid(
    aggregate_run_id: uuid.UUID,
    llm_call_ids: tuple[uuid.UUID, ...]
) -> uuid.UUID:
    # Sort for deterministic ordering (set of calls, not ordered list)
    sorted_call_ids = ":".join(str(cid) for cid in sorted(llm_call_ids))
    natural_key = f"{aggregate_run_id}:{sorted_call_ids}"
    return uuid.uuid5(NAMESPACE_AGGREGATED_SCORE, natural_key)


def compute_aggregated_score_llm_call_uuid(
    aggregated_score_id: uuid.UUID,
    llm_call_id: uuid.UUID
) -> uuid.UUID:
    natural_key = f"{aggregated_score_id}:{llm_call_id}"
    return uuid.uuid5(NAMESPACE_AGGREGATED_SCORE_LLM_CALL, natural_key)
