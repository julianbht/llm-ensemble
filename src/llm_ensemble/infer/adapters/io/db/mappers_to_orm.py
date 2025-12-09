"""Domain to ORM mappers for INFER CLI (writing direction).

This module provides conversion functions for mapping from Pydantic domain objects
to SQLAlchemy ORM models for persistence.

Design principles:
- Domain objects are the source of truth (LLMJudgement, ModelConfig, PromptConfig)
- Mappers convert domain → ORMs for SQL persistence (INSERT operations)
- Mappers compute UUIDs based on ORM natural keys
- Stateless pure functions
- Used by SqlJudgementWriter for persisting inference results

The domain layer works with Pydantic models (LLMJudgement, configs).
The persistence layer works with SQLAlchemy ORMs.
These mappers handle the impedance mismatch for the write path.
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.entities.llm_judgement import (
    LLMJudgement,
)
from llm_ensemble.infer.schemas.entities.llm_prompt import LLMPrompt
from llm_ensemble.infer.schemas.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.schemas.entities.llm_score import LLMScore
from llm_ensemble.infer.adapters.io.db.orms import (
    ProviderORM,
    ModelConfigORM,
    PromptTemplateORM,
    ParserORM,
    InferRunORM,
    LLMPromptORM,
    LLMResponseTextORM,
    LLMInvocationMetricsORM,
    LLMScoreORM,
    LLMJudgementORM,
    JudgedDatasetORM,
)
from llm_ensemble.libs.db import (
    compute_judged_dataset_uuid,
    compute_infer_run_uuid,
)
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


# ============================================================================
# Provider Mappers
# ============================================================================

def provider_to_orm(provider: "Provider") -> ProviderORM:
    """Convert Provider domain object to ProviderORM.

    Args:
        provider: Provider domain object (already has random UUID)

    Returns:
        ProviderORM model ready for persistence
    """
    return ProviderORM(
        id=provider.id,
        name=provider.name,
    )


# ============================================================================
# ModelConfig Mappers
# ============================================================================

def model_config_to_orm(model_cfg: ModelConfig) -> ModelConfigORM:
    """Convert ModelConfig to ModelConfigORM.

    Collapses model identity, capabilities, and inference params into single ORM.
    No separate Model entity - ModelConfig is the complete configuration.
    Provider is tracked on JudgedDataset (runtime fact), not here (configuration).

    Args:
        model_cfg: ModelConfig object

    Returns:
        ModelConfigORM model ready for persistence
    """
    # Prepare additional_params (catch-all for non-explicit fields)
    additional_params = model_cfg.model_specs.additional_params.copy() if model_cfg.model_specs.additional_params else {}
    if model_cfg.model_specs.stop:
        additional_params["stop"] = model_cfg.model_specs.stop
    if model_cfg.model_specs.response_format:
        additional_params["response_format"] = model_cfg.model_specs.response_format

    return ModelConfigORM(
        id=model_cfg.id,
        name=model_cfg.name_hint,
        model_id=model_cfg.model_name,
        context_window=model_cfg.model_specs.context_window,
        capabilities=model_cfg.model_specs.capabilities if model_cfg.model_specs.capabilities else None,
        temperature=model_cfg.model_specs.temperature,
        max_tokens=model_cfg.model_specs.max_tokens,
        top_p=model_cfg.model_specs.top_p,
        frequency_penalty=model_cfg.model_specs.frequency_penalty,
        presence_penalty=model_cfg.model_specs.presence_penalty,
        seed=model_cfg.model_specs.seed,
        additional_params=additional_params if additional_params else None,
    )


# ============================================================================
# PromptTemplate Mappers
# ============================================================================

def prompt_template_to_orm(prompt_template: "PromptTemplate") -> PromptTemplateORM:
    """Convert PromptTemplate domain object to PromptTemplateORM.

    Args:
        prompt_template: PromptTemplate domain object (already has random UUID)

    Returns:
        PromptTemplateORM model ready for persistence
    """
    return PromptTemplateORM(
        id=prompt_template.id,
        name=prompt_template.name,
        template_text=prompt_template.template_text,
    )


# ============================================================================
# ParserSpec Mappers
# ============================================================================

def parser_to_orm(parser: "Parser") -> ParserORM:
    """Convert Parser domain object to ParserORM.

    Args:
        parser: Parser domain object (already has random UUID)

    Returns:
        ParserORM model ready for persistence
    """
    return ParserORM(
        id=parser.id,
        name=parser.name,
    )


# ============================================================================
# InferRun Mappers
# ============================================================================

def infer_run_info_to_orm(
    run_info: InferRunInfo,
    config_names: dict[str, str],
    start_idx: int,
    end_idx: int,
) -> InferRunORM:
    """Convert InferRunInfo to InferRunORM.

    Args:
        run_info: InferRunInfo context object
        config_names: Config names dict {model_config, prompt_name, parser_name}
        start_idx: Computed start index into NormalizedDataset.samples
        end_idx: Computed end index into NormalizedDataset.samples

    Returns:
        InferRunORM model ready for persistence (without judged_dataset_id)
    """
    return InferRunORM(
        id=run_info.id,
        run_name=run_info.run_name,
        run_type=run_info.run_type,
        config_names=config_names,
        start_idx=start_idx,
        end_idx=end_idx,
        judged_dataset_id=None,  # Set in close() after computing actual dataset
        git_sha=run_info.git_sha,
        git_branch=run_info.git_branch,
        git_is_dirty=not run_info.git_clean,
        notes=run_info.notes,
    )


# ============================================================================
# LLMPromptText Mappers
# ============================================================================

def llm_prompt_to_orm(
    llm_prompt: LLMPrompt,
    prompt_template_id: UUID,
    dataset_sample_id: UUID,
) -> LLMPromptORM:
    """Convert LLMPrompt domain object to LLMPromptTextORM.

    UUID is computed from (prompt_template_id, dataset_sample_id, prompt_text).

    Args:
        llm_prompt: LLMPrompt domain object
        prompt_template_id: PromptTemplate UUID (for foreign key)
        dataset_sample_id: DatasetSample UUID (cross-schema reference to ingest.dataset_sample)

    Returns:
        LLMPromptTextORM model ready for persistence
    """
    import hashlib
    prompt_hash = hashlib.sha256(llm_prompt.prompt_text.encode()).hexdigest()

    # Natural key: (prompt_template_id, dataset_sample_id, prompt_text)
    from uuid import uuid5, NAMESPACE_DNS
    prompt_id = uuid5(
        NAMESPACE_DNS,
        f"{prompt_template_id}:{dataset_sample_id}:{prompt_hash}"
    )

    return LLMPromptORM(
        id=prompt_id,
        prompt_template_id=prompt_template_id,
        dataset_sample_id=dataset_sample_id,
        prompt_text=llm_prompt.prompt_text,
    )


# ============================================================================
# LLMResponseText Mappers
# ============================================================================

def llm_response_text_to_orm(llm_response_text: str) -> LLMResponseTextORM:
    """Convert raw LLM response text to LLMResponseTextORM.

    UUID is computed from llm_response_text content hash.

    Args:
        llm_response_text: Raw LLM response text string

    Returns:
        LLMResponseTextORM model ready for persistence
    """
    response_id = compute_llm_response_text_uuid(llm_response_text)
    return LLMResponseTextORM(
        id=response_id,
        llm_response_text=llm_response_text,
    )


# ============================================================================
# LLMInvocationMetrics Mappers
# ============================================================================

def llm_invocation_metrics_to_orm(metrics: LLMInvocationMetrics) -> LLMInvocationMetricsORM:
    """Convert LLMInvocationMetrics domain object to LLMInvocationMetricsORM.

    UUID is computed from all metric fields.

    Args:
        metrics: LLMInvocationMetrics domain object

    Returns:
        LLMInvocationMetricsORM model ready for persistence
    """
    metrics_id = compute_llm_invocation_metrics_uuid(
        latency_ms=metrics.latency_ms,
        retries=metrics.retries,
        cost_estimate_usd=metrics.cost_estimate_usd,
        generation_id=metrics.generation_id,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
    )
    return LLMInvocationMetricsORM(
        id=metrics_id,
        latency_ms=metrics.latency_ms,
        retries=metrics.retries,
        cost_estimate_usd=metrics.cost_estimate_usd,
        generation_id=metrics.generation_id,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
    )


# ============================================================================
# LLMScore Mappers
# ============================================================================

def llm_score_to_orm(
    llm_score: LLMScore,
    parser_spec_id: UUID,
    llm_response_text_id: UUID,
) -> LLMScoreORM:
    """Convert LLMScore domain object to LLMScoreORM.

    UUID is computed from (parser_spec_id, llm_response_text_id).

    Args:
        llm_score: LLMScore domain object
        parser_spec_id: Parser spec UUID (for foreign key)
        llm_response_text_id: LLM response text UUID (for foreign key)

    Returns:
        LLMScoreORM model ready for persistence
    """
    score_id = compute_llm_score_uuid(parser_spec_id, llm_response_text_id)

    # Handle case where label is None (parsing failed) - need default for non-nullable column
    label = llm_score.label if llm_score.label is not None else RelevanceScore.NOT_RELEVANT

    # Extract parser warnings (convert to dict format for JSONB array)
    parser_warnings = [w.to_dict() for w in llm_score.warnings] if llm_score.warnings else []

    return LLMScoreORM(
        id=score_id,
        parser_spec_id=parser_spec_id,
        llm_response_text_id=llm_response_text_id,
        label=label,
        confidence=llm_score.confidence,
        rationale=llm_score.rationale,
        parser_warnings=parser_warnings,
    )


# ============================================================================
# LLMJudgement Mappers
# ============================================================================

def llm_judgement_to_orm(
    judgement: LLMJudgement,
    judged_dataset_id: UUID,
    llm_prompt_text_id: UUID,
    llm_invocation_metrics_id: UUID,
    llm_score_id: UUID,
) -> LLMJudgementORM:
    """Convert LLMJudgement domain object to LLMJudgementORM.

    UUID is computed from natural key (judged_dataset_id, llm_prompt_text_id).

    Args:
        judgement: LLMJudgement domain object
        judged_dataset_id: JudgedDataset UUID (for foreign key)
        llm_prompt_text_id: LLMPromptText UUID (for foreign key)
        llm_invocation_metrics_id: LLMInvocationMetrics UUID (for foreign key)
        llm_score_id: LLMScore UUID (for foreign key)

    Returns:
        LLMJudgementORM model ready for persistence
    """
    judgement_id = compute_llm_judgement_uuid(judged_dataset_id, llm_prompt_text_id)

    return LLMJudgementORM(
        id=judgement_id,
        judged_dataset_id=judged_dataset_id,
        llm_prompt_text_id=llm_prompt_text_id,
        llm_invocation_metrics_id=llm_invocation_metrics_id,
        llm_score_id=llm_score_id,
    )


# ============================================================================
# JudgedDataset Mappers
# ============================================================================

def judged_dataset_to_orm(fingerprint: str) -> JudgedDatasetORM:
    """Create JudgedDatasetORM.

    UUID is computed from fingerprint (SHA256 of sorted judgement IDs).

    Args:
        fingerprint: SHA256 hash of sorted judgement IDs

    Returns:
        JudgedDatasetORM model ready for persistence
    """
    dataset_id = compute_judged_dataset_uuid(fingerprint)
    return JudgedDatasetORM(
        id=dataset_id,
        fingerprint=fingerprint,
    )
