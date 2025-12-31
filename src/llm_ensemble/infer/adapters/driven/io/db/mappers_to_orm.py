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
from typing import Optional
from uuid import UUID

from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.infer_run import InferRun
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    ProviderORM,
    ModelConfigORM,
    PromptBuilderORM,
    ParserORM,
    PromptTemplateORM,
    InferRunConfigORM,
    InferRunORM,
    LLMPromptTextORM,
    LLMResponseTextORM,
    LLMScoreORM,
    LLMJudgementORM,
    InferRunOutputORM,
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
        version=provider.version,
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
        name_hint=model_cfg.name_hint,
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
        version=prompt_builder.version,
    )


# ============================================================================
# Parser Mappers
# ============================================================================

def parser_to_orm(parser: ResponseParser) -> ParserORM:
    """Convert Parser domain object to ParserORM.

    Args:
        parser: Parser domain object (already has UUID)

    Returns:
        ParserORM model ready for persistence
    """
    return ParserORM(
        id=parser.id,
        name=parser.name,
        version=parser.version,
    )


# ============================================================================
# PromptTemplate Mappers
# ============================================================================

def prompt_template_to_orm(prompt_template: PromptTemplate) -> PromptTemplateORM:
    """Convert PromptTemplate domain object to PromptTemplateORM.

    Args:
        prompt_template: PromptTemplate domain object (already has UUID)

    Returns:
        PromptTemplateORM model ready for persistence
    """
    return PromptTemplateORM(
        id=prompt_template.id,
        name=prompt_template.name,
        template_text=prompt_template.template_text,
        prompt_builder_id=prompt_template.prompt_builder.id,
        parser_id=prompt_template.response_parser.id,
    )


# ============================================================================
# InferRunConfig Mappers
# ============================================================================

def infer_run_config_to_orm(infer_run_config: InferRunConfig) -> InferRunConfigORM:
    """Convert InferRunConfig domain object to InferRunConfigORM.

    Execution context fields (input_run_name, start_idx, end_idx) are inlined.

    Args:
        infer_run_config: InferRunConfig domain object (already has UUID)

    Returns:
        InferRunConfigORM model ready for persistence
    """
    return InferRunConfigORM(
        id=infer_run_config.id,
        model_config_id=infer_run_config.model_cfg.id,
        provider_id=infer_run_config.provider.id,
        prompt_template_id=infer_run_config.prompt_template.id,
        input_run_name=infer_run_config.input_run_name,
        start_idx=infer_run_config.start_idx,
        end_idx=infer_run_config.end_idx,
        io_name=infer_run_config.io_name,
    )


# ============================================================================
# InferRun Mappers
# ============================================================================

def infer_run_to_orm(
    infer_run: InferRun,
    infer_run_config_id: UUID,
    infer_run_output_id: Optional[UUID] = None,
) -> InferRunORM:
    """Convert InferRun aggregate root to InferRunORM.

    Args:
        infer_run: InferRun aggregate root (already has UUID)
        infer_run_config_id: InferRunConfig UUID (for foreign key)
        infer_run_output_id: InferRunOutput UUID (for foreign key, None until run completes)

    Returns:
        InferRunORM model ready for persistence
    """
    return InferRunORM(
        id=infer_run.id,
        run_name=infer_run.run_name,
        run_type=infer_run.run_type,
        infer_run_config_id=infer_run_config_id,
        infer_run_output_id=infer_run_output_id,
        git_sha=infer_run.git_info.git_sha,
        git_branch=infer_run.git_info.git_branch,
        git_is_dirty=not infer_run.git_info.git_clean,
        notes=infer_run.notes,
        start_time=infer_run.start_time,
        end_time=infer_run.end_time,
    )


# ============================================================================
# LLMPromptText Mappers
# ============================================================================

def llm_prompt_text_to_orm(
    prompt_text: str,
    content_hash: str,
    prompt_text_id: UUID,
) -> LLMPromptTextORM:
    """Convert prompt text string to LLMPromptTextORM.

    Args:
        prompt_text: Rendered prompt text string
        content_hash: SHA256 hash of prompt text for deduplication
        prompt_text_id: Pre-computed UUID for this prompt text (from deduplication logic)

    Returns:
        LLMPromptTextORM model ready for persistence
    """
    return LLMPromptTextORM(
        id=prompt_text_id,
        content_hash=content_hash,
        prompt_text=prompt_text,
    )


# ============================================================================
# LLMResponseText Mappers
# ============================================================================

def llm_response_text_to_orm(
    response_text: str,
    content_hash: str,
    response_text_id: UUID,
) -> LLMResponseTextORM:
    """Convert raw LLM response text to LLMResponseTextORM.

    Args:
        response_text: Raw LLM response text string
        content_hash: SHA256 hash of response text for deduplication
        response_text_id: Pre-computed UUID for this response text (from deduplication logic)

    Returns:
        LLMResponseTextORM model ready for persistence
    """
    return LLMResponseTextORM(
        id=response_text_id,
        content_hash=content_hash,
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
    infer_run_output_id: UUID,
    dataset_sample_id: UUID,
    llm_prompt_text_id: UUID,
    llm_response_text_id: UUID,
    llm_score_id: UUID,
) -> LLMJudgementORM:
    """Convert LLMJudgement domain object to LLMJudgementORM.

    Inlines metrics and warnings directly on the judgement ORM.

    Args:
        judgement: LLMJudgement domain object (already has UUID)
        infer_run_output_id: InferRunOutput UUID (for foreign key)
        dataset_sample_id: DatasetSample UUID (cross-schema reference)
        llm_prompt_text_id: LLMPromptText UUID (for foreign key)
        llm_response_text_id: LLMResponseText UUID (for foreign key)
        llm_score_id: LLMScore UUID (for foreign key)

    Returns:
        LLMJudgementORM model ready for persistence
    """
    # Extract parser warnings from judgement (not from llm_score)
    parse_issues = [w.to_dict() for w in judgement.parse_issues] if judgement.parse_issues else []

    return LLMJudgementORM(
        id=judgement.id,
        infer_run_output_id=infer_run_output_id,
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
        parse_issues=parse_issues,
    )


# ============================================================================
# InferRunOutput Mappers
# ============================================================================

def infer_run_output_to_orm(
    infer_run_output: InferRunOutput,
) -> InferRunOutputORM:
    """Convert InferRunOutput to InferRunOutputORM.

    Configuration reference is via InferRunORM (not stored here).

    Args:
        infer_run_output: InferRunOutput domain object (already has UUID, same as InferRun.id)

    Returns:
        InferRunOutputORM model ready for persistence
    """
    return InferRunOutputORM(
        id=infer_run_output.id,
        sample_fingerprint=infer_run_output.sample_fingerprint,
    )
