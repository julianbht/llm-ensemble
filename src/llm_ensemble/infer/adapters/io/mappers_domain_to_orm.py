"""Domain to ORM mappers for INFER CLI (writing direction).

This module provides conversion functions for mapping from Pydantic domain objects
to SQLAlchemy ORM models for persistence.

Design principles:
- Domain objects are the source of truth (LLMJudgement, ModelConfig, PromptConfig)
- Mappers convert domain → ORMs for SQL persistence (INSERT operations)
- Stateless pure functions
- Used by SqlJudgementWriter for persisting inference results

The domain layer works with Pydantic models (LLMJudgement, configs).
The persistence layer works with SQLAlchemy ORMs (LLMCallORM, etc.).
These mappers handle the impedance mismatch for the write path.
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.orms_normalized import (
    ProviderORM,
    ModelSpecORM,
    PromptTemplateORM,
    ParserSpecORM,
    InferredDatasetORM,
    InferRunORM,
    LLMRequestORM,
    LLMScoreORM,
    LLMCallORM,
)
from llm_ensemble.libs.db import (
    compute_provider_uuid,
    compute_model_spec_uuid,
    compute_prompt_template_uuid,
    compute_parser_spec_uuid,
    compute_ingest_run_uuid,
)
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


# ============================================================================
# Provider Mappers
# ============================================================================

def provider_name_to_orm(provider_name: str) -> ProviderORM:
    """Convert provider name string to ProviderORM.

    Args:
        provider_name: Provider name (e.g., 'openrouter', 'ollama', 'hf')

    Returns:
        ProviderORM model ready for persistence
    """
    provider_id = compute_provider_uuid(provider_name)
    return ProviderORM(
        id=provider_id,
        name=provider_name,
    )


# ============================================================================
# ModelSpec Mappers
# ============================================================================

def model_config_to_orm(model_cfg: ModelConfig, provider_id: UUID) -> ModelSpecORM:
    """Convert ModelConfig to ModelSpecORM.

    Note: provider_id must be provided explicitly as the ORM needs the foreign key.
    The ModelConfig has the provider name, which we use to compute the provider_id.

    Args:
        model_cfg: ModelConfig object
        provider_id: Provider UUID (for foreign key)

    Returns:
        ModelSpecORM model ready for persistence
    """
    # Prepare additional_params (catch-all for non-explicit fields)
    additional_params = model_cfg.additional_params.copy() if model_cfg.additional_params else {}
    if model_cfg.stop:
        additional_params["stop"] = model_cfg.stop
    if model_cfg.response_format:
        additional_params["response_format"] = model_cfg.response_format

    return ModelSpecORM(
        id=compute_model_spec_uuid(model_cfg.name),
        name=model_cfg.name,
        model_id=model_cfg.model_id,
        provider_id=provider_id,
        context_window=model_cfg.context_window,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        top_p=model_cfg.top_p,
        frequency_penalty=model_cfg.frequency_penalty,
        presence_penalty=model_cfg.presence_penalty,
        seed=model_cfg.seed,
        additional_params=additional_params if additional_params else None,
        capabilities=model_cfg.capabilities if model_cfg.capabilities else None,
    )


# ============================================================================
# PromptTemplate Mappers
# ============================================================================

def prompt_config_to_template_orm(prompt_cfg: PromptConfig, template_text: str) -> PromptTemplateORM:
    """Convert PromptConfig to PromptTemplateORM.

    Note: template_text must be provided explicitly as it's loaded from the builder.
    The PromptConfig knows how to get the builder, but the mapper is stateless.

    Args:
        prompt_cfg: PromptConfig object
        template_text: Template text (loaded from builder)

    Returns:
        PromptTemplateORM model ready for persistence
    """
    return PromptTemplateORM(
        id=compute_prompt_template_uuid(prompt_cfg.name),
        name=prompt_cfg.name,
        template_text=template_text,
    )


# ============================================================================
# ParserSpec Mappers
# ============================================================================

def prompt_config_to_parser_orm(prompt_cfg: PromptConfig, code_hash: str) -> ParserSpecORM:
    """Convert PromptConfig to ParserSpecORM.

    Note: code_hash must be provided explicitly. Using placeholder for now.

    Args:
        prompt_cfg: PromptConfig object
        code_hash: Parser code hash (for versioning)

    Returns:
        ParserSpecORM model ready for persistence
    """
    return ParserSpecORM(
        id=compute_parser_spec_uuid(
            prompt_cfg.parser_module,
            prompt_cfg.parser_class,
            code_hash
        ),
        code_hash=code_hash,
        parser_module=prompt_cfg.parser_module,
        parser_class=prompt_cfg.parser_class,
    )


# ============================================================================
# InferRun Mappers
# ============================================================================

def infer_run_info_to_orm(
    run_info: InferRunInfo,
    model_spec_id: UUID,
    prompt_template_id: UUID,
    parser_spec_id: UUID,
) -> InferRunORM:
    """Convert InferRunInfo to InferRunORM.

    Note: Foreign key IDs must be provided explicitly as they're derived from
    the related entities (ModelSpec, PromptTemplate, ParserSpec, InferredDataset).

    The ingest_run_id is computed deterministically from the input_run_name.

    Args:
        run_info: InferRunInfo context object
        model_spec_id: ModelSpec UUID (for foreign key)
        prompt_template_id: PromptTemplate UUID (for foreign key)
        parser_spec_id: ParserSpec UUID (for foreign key)

    Returns:
        InferRunORM model ready for persistence
    """
    # Compute ingest run ID from input run name (deterministic UUID)
    ingest_run_id = compute_ingest_run_uuid(run_info.input_run_name)

    return InferRunORM(
        id=run_info.id,
        run_name=run_info.run_name,
        run_type=run_info.run_type,
        model_spec_id=model_spec_id,
        prompt_template_id=prompt_template_id,
        parser_spec_id=parser_spec_id,
        ingest_run_id=ingest_run_id,
        limit=run_info.limit,
        git_sha=run_info.git_sha,
        git_branch=run_info.git_branch,
        git_is_dirty=not run_info.git_clean,
        notes=run_info.notes,
    )


# ============================================================================
# LLMRequest Mappers
# ============================================================================

def llm_judgement_to_request_orm(judgement: LLMJudgement, request_id: UUID) -> LLMRequestORM:
    """Convert LLMJudgement to LLMRequestORM.

    Args:
        judgement: LLMJudgement domain object
        request_id: Pre-computed request UUID

    Returns:
        LLMRequestORM model ready for persistence
    """
    return LLMRequestORM(
        id=request_id,
        judging_sample_id=judgement.judging_sample.id,
        prompt=judgement.prompt,
    )


# ============================================================================
# LLMScore Mappers
# ============================================================================

def llm_judgement_to_score_orm(
    judgement: LLMJudgement,
    score_id: UUID,
    parser_spec_id: UUID
) -> LLMScoreORM:
    """Convert LLMJudgement to LLMScoreORM.

    Args:
        judgement: LLMJudgement domain object
        score_id: Pre-computed score UUID
        parser_spec_id: Parser spec UUID (for foreign key)

    Returns:
        LLMScoreORM model ready for persistence
    """
    # Extract parsed fields from llm_score (may be None if parsing failed)
    label = judgement.llm_score.label if judgement.llm_score else None
    confidence = judgement.llm_score.confidence if judgement.llm_score else None
    rationale = judgement.llm_score.rationale if judgement.llm_score else None

    # Extract parser warnings (convert to dict format for JSONB array)
    parser_warnings = []
    if judgement.llm_score:
        parser_warnings = [w.to_dict() for w in judgement.llm_score.warnings]

    # Handle case where label is None (parsing failed) - need default for non-nullable column
    if label is None:
        label = RelevanceScore.NOT_RELEVANT

    return LLMScoreORM(
        id=score_id,
        parser_spec_id=parser_spec_id,
        raw_response=judgement.llm_response.raw_response,
        label=label,
        confidence=confidence,
        rationale=rationale,
        parser_warnings=parser_warnings,
    )


# ============================================================================
# LLMCall Mappers
# ============================================================================

def llm_judgement_to_call_orm(
    judgement: LLMJudgement,
    call_id: UUID,
    request_id: UUID,
    infer_run_id: UUID,
    score_id: UUID
) -> LLMCallORM:
    """Convert LLMJudgement to LLMCallORM.

    Args:
        judgement: LLMJudgement domain object
        call_id: Pre-computed call UUID
        request_id: Request UUID (for foreign key)
        infer_run_id: Infer run UUID (for foreign key)
        score_id: Score UUID (for foreign key)

    Returns:
        LLMCallORM model ready for persistence
    """
    return LLMCallORM(
        id=call_id,
        llm_request_id=request_id,
        infer_run_id=infer_run_id,
        score_id=score_id,
        latency_ms=judgement.llm_response.latency_ms,
        retries=judgement.llm_response.retries,
        cost_estimate_usd=judgement.llm_response.cost_estimate_usd,
        generation_id=judgement.llm_response.generation_id,
        prompt_tokens=judgement.llm_response.prompt_tokens,
        completion_tokens=judgement.llm_response.completion_tokens,
        total_tokens=judgement.llm_response.total_tokens,
    )
