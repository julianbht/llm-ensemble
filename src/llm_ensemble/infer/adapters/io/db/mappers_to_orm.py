"""Domain to ORM mappers for INFER CLI (writing direction).

This module provides conversion functions for mapping from Pydantic domain objects
to SQLAlchemy ORM models for persistence.

Design principles:
- Domain objects are the source of truth and already have UUIDs
- Mappers are simple pass-through converters (no UUID computation)
- Stateless pure functions
- Used by DBWriter for persisting inference results

The domain layer works with Pydantic models (LLMJudgement, configs).
The persistence layer works with SQLAlchemy ORMs.
These mappers handle the impedance mismatch for the write path.
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.entities.llm_score import LLMScore
from llm_ensemble.infer.schemas.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.schemas.entities.parser import Parser
from llm_ensemble.infer.schemas.entities.provider import Provider
from llm_ensemble.infer.schemas.entities.adapter_config import AdapterConfig
from llm_ensemble.infer.adapters.io.db.orms import (
    ProviderORM,
    ModelConfigORM,
    PromptBuilderORM,
    ParserORM,
    AdapterConfigORM,
    InferRunORM,
    LLMPromptTextORM,
    LLMResponseTextORM,
    LLMScoreORM,
    LLMJudgementORM,
    JudgedDatasetORM,
)
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


# ============================================================================
# Provider Mappers
# ============================================================================

def provider_to_orm(provider: Provider) -> ProviderORM:
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

    Direct 1:1 mapping from flat domain object to flat ORM.

    Args:
        model_cfg: ModelConfig object (already has random UUID)

    Returns:
        ModelConfigORM model ready for persistence
    """
    return ModelConfigORM(
        id=model_cfg.id,
        name=model_cfg.name,
        model_id=model_cfg.model_id,
        context_window=model_cfg.context_window,
        capabilities=model_cfg.capabilities,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        top_p=model_cfg.top_p,
        frequency_penalty=model_cfg.frequency_penalty,
        presence_penalty=model_cfg.presence_penalty,
        seed=model_cfg.seed,
        additional_params=model_cfg.additional_params,
    )


# ============================================================================
# PromptBuilder Mappers
# ============================================================================

def prompt_builder_to_orm(prompt_builder: PromptBuilder) -> PromptBuilderORM:
    """Convert PromptBuilder domain object to PromptBuilderORM.

    Args:
        prompt_builder: PromptBuilder domain object (already has UUID)

    Returns:
        PromptBuilderORM model ready for persistence
    """
    return PromptBuilderORM(
        id=prompt_builder.id,
        name=prompt_builder.name,
        template_text=prompt_builder.template_text,
    )


# ============================================================================
# Parser Mappers
# ============================================================================

def parser_to_orm(parser: Parser) -> ParserORM:
    """Convert Parser domain object to ParserORM.

    Args:
        parser: Parser domain object (already has UUID)

    Returns:
        ParserORM model ready for persistence
    """
    return ParserORM(
        id=parser.id,
        name=parser.name,
    )


# ============================================================================
# AdapterConfig Mappers
# ============================================================================

def adapter_config_to_orm(adapter_config: AdapterConfig) -> AdapterConfigORM:
    """Convert AdapterConfig domain object to AdapterConfigORM.

    Args:
        adapter_config: AdapterConfig domain object (already has UUID)

    Returns:
        AdapterConfigORM model ready for persistence
    """
    return AdapterConfigORM(
        id=adapter_config.id,
        prompt_builder_id=adapter_config.prompt_builder.id,
        parser_id=adapter_config.parser.id,
        provider_id=adapter_config.provider.id,
    )


# ============================================================================
# InferRun Mappers
# ============================================================================

def infer_run_info_to_orm(
    run_info: InferRunInfo,
    start_idx: int,
    end_idx: int,
) -> InferRunORM:
    """Convert InferRunInfo to InferRunORM.

    Args:
        run_info: InferRunInfo context object (already has UUID)
        start_idx: Computed start index into NormalizedDataset.samples
        end_idx: Computed end index into NormalizedDataset.samples

    Returns:
        InferRunORM model ready for persistence (without judged_dataset_id)
    """
    return InferRunORM(
        id=run_info.id,
        run_name=run_info.run_name,
        run_type=run_info.run_type,
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

def llm_prompt_text_to_orm(
    prompt_text: str,
    prompt_text_id: UUID,
) -> LLMPromptTextORM:
    """Convert prompt text string to LLMPromptTextORM.

    Args:
        prompt_text: Rendered prompt text string
        prompt_text_id: Pre-computed UUID for this prompt text (from deduplication logic)

    Returns:
        LLMPromptTextORM model ready for persistence
    """
    return LLMPromptTextORM(
        id=prompt_text_id,
        prompt_text=prompt_text,
    )


# ============================================================================
# LLMResponseText Mappers
# ============================================================================

def llm_response_text_to_orm(
    response_text: str,
    response_text_id: UUID,
) -> LLMResponseTextORM:
    """Convert raw LLM response text to LLMResponseTextORM.

    Args:
        response_text: Raw LLM response text string
        response_text_id: Pre-computed UUID for this response text (from deduplication logic)

    Returns:
        LLMResponseTextORM model ready for persistence
    """
    return LLMResponseTextORM(
        id=response_text_id,
        llm_response_text=response_text,
    )


# ============================================================================
# LLMScore Mappers
# ============================================================================

def llm_score_to_orm(
    llm_score: LLMScore,
) -> LLMScoreORM:
    """Convert LLMScore domain object to LLMScoreORM.

    Args:
        llm_score: LLMScore domain object (already has UUID)

    Returns:
        LLMScoreORM model ready for persistence
    """
    # Handle case where label is None (parsing failed) - need default for non-nullable column
    label = llm_score.label if llm_score.label is not None else RelevanceScore.NOT_RELEVANT

    # Extract parser warnings (convert to dict format for JSONB array)
    parser_warnings = [w.to_dict() for w in llm_score.warnings] if llm_score.warnings else []

    return LLMScoreORM(
        id=llm_score.id,
        label=label,
        confidence=llm_score.confidence,
        rationale=llm_score.rationale,
    )


# ============================================================================
# LLMJudgement Mappers
# ============================================================================

def llm_judgement_to_orm(
    judgement: LLMJudgement,
    judged_dataset_id: UUID,
    dataset_sample_id: UUID,
    llm_prompt_text_id: UUID,
    llm_response_text_id: UUID,
    llm_score_id: UUID,
) -> LLMJudgementORM:
    """Convert LLMJudgement domain object to LLMJudgementORM.

    Inlines metrics and warnings directly on the judgement ORM.

    Args:
        judgement: LLMJudgement domain object (already has UUID)
        judged_dataset_id: JudgedDataset UUID (for foreign key)
        dataset_sample_id: DatasetSample UUID (cross-schema reference)
        llm_prompt_text_id: LLMPromptText UUID (for foreign key)
        llm_response_text_id: LLMResponseText UUID (for foreign key)
        llm_score_id: LLMScore UUID (for foreign key)

    Returns:
        LLMJudgementORM model ready for persistence
    """
    # Extract parser warnings from judgement (not from llm_score)
    parser_warnings = [w.to_dict() for w in judgement.parser_warnings] if judgement.parser_warnings else []

    return LLMJudgementORM(
        id=judgement.id,
        judged_dataset_id=judged_dataset_id,
        dataset_sample_id=dataset_sample_id,
        llm_prompt_text_id=llm_prompt_text_id,
        llm_response_text_id=llm_response_text_id,
        llm_score_id=llm_score_id,
        # Inline invocation metrics (no longer separate table)
        latency_ms=judgement.llm_invocation_metrics.latency_ms,
        retries=judgement.llm_invocation_metrics.retries,
        cost_estimate_usd=judgement.llm_invocation_metrics.cost_estimate_usd,
        generation_id=judgement.llm_invocation_metrics.generation_id,
        prompt_tokens=judgement.llm_invocation_metrics.prompt_tokens,
        completion_tokens=judgement.llm_invocation_metrics.completion_tokens,
        total_tokens=judgement.llm_invocation_metrics.total_tokens,
        parser_warnings=parser_warnings,
    )


# ============================================================================
# JudgedDataset Mappers
# ============================================================================

def judged_dataset_to_orm(
    judged_dataset_id: UUID,
    model_config_id: UUID,
    adapter_config_id: UUID,
    sample_fingerprint: str,
) -> JudgedDatasetORM:
    """Create JudgedDatasetORM.

    Args:
        judged_dataset_id: JudgedDataset UUID (same as InferRun.id for 1:1 relationship)
        model_config_id: ModelConfig UUID (for foreign key)
        adapter_config_id: AdapterConfig UUID (for foreign key)
        sample_fingerprint: SHA256 hash of sorted dataset_sample IDs

    Returns:
        JudgedDatasetORM model ready for persistence
    """
    return JudgedDatasetORM(
        id=judged_dataset_id,
        model_config_id=model_config_id,
        adapter_config_id=adapter_config_id,
        sample_fingerprint=sample_fingerprint,
    )
