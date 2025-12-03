"""Domain to ORM mappers for AGGREGATE CLI (writing direction).

This module provides conversion functions for mapping from Pydantic domain objects
to SQLAlchemy ORM models for persistence.

Design principles:
- Domain objects are the source of truth (AggregatedDataset, AggregatedVote, etc.)
- Mappers convert domain → ORMs for SQL persistence (INSERT operations)
- UUIDs are already computed in domain objects (do NOT recompute)
- Stateless pure functions
- Used by SqlAggregatedDatasetWriter for persisting aggregation results

The domain layer works with Pydantic models.
The persistence layer works with SQLAlchemy ORMs.
These mappers handle the impedance mismatch for the write path.
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.aggregate.schemas.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.schemas.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.schemas.aggregated_dataset import AggregatedDataset
from llm_ensemble.aggregate.schemas.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.schemas.orms_normalized import (
    AggregationStrategyORM,
    AggregateRunORM,
    AggregatedDatasetORM,
    AggregatedVoteORM,
    AggregationVoteORM,
    AggregatedDatasetVoteORM,
)


# ============================================================================
# AggregationStrategy Mappers
# ============================================================================

def aggregation_strategy_to_orm(
    aggregation_strategy: AggregationStrategy,
) -> AggregationStrategyORM:
    """Convert AggregationStrategy entity to AggregationStrategyORM.

    UUID is already computed in the entity.

    Args:
        aggregation_strategy: AggregationStrategy domain entity

    Returns:
        AggregationStrategyORM model ready for persistence (minimal entity: just id + name)
    """
    return AggregationStrategyORM(
        id=aggregation_strategy.id,
        name=aggregation_strategy.name,
    )


# ============================================================================
# AggregateRun Mappers
# ============================================================================

def aggregate_run_info_to_orm(
    run_info: AggregateRunInfo,
    aggregate_run_id: UUID,
    config_names: dict[str, str],
) -> AggregateRunORM:
    """Convert AggregateRunInfo to AggregateRunORM.

    Args:
        run_info: AggregateRunInfo domain object
        aggregate_run_id: Pre-computed UUID for this run
        config_names: Dict of config names (e.g., {"aggregation_strategy": "majority_vote", "io_config": "db"})

    Returns:
        AggregateRunORM model ready for persistence
    """
    return AggregateRunORM(
        id=aggregate_run_id,
        run_name=run_info.run_name,
        run_type=run_info.run_type,
        config_names=config_names,
        aggregated_dataset_id=None,  # Set in close() after dataset is created
        git_sha=run_info.git_sha,
        git_branch=run_info.git_branch,
        git_is_dirty=run_info.git_clean,
        notes=run_info.notes,
    )


# ============================================================================
# AggregatedDataset Mappers
# ============================================================================

def aggregated_dataset_to_orm(
    aggregated_dataset: AggregatedDataset,
) -> AggregatedDatasetORM:
    """Convert AggregatedDataset to AggregatedDatasetORM.

    UUID and fingerprint are already computed in the domain object.

    Args:
        aggregated_dataset: AggregatedDataset domain object

    Returns:
        AggregatedDatasetORM model ready for persistence
    """
    return AggregatedDatasetORM(
        id=aggregated_dataset.id,
        fingerprint=aggregated_dataset.fingerprint,
    )


# ============================================================================
# AggregatedVote Mappers
# ============================================================================

def aggregated_vote_to_orm(
    aggregated_vote: AggregatedVote,
) -> AggregatedVoteORM:
    """Convert AggregatedVote to AggregatedVoteORM.

    UUID is already computed in the domain object.
    Extract dataset_sample_id from first llm_judgement.

    Args:
        aggregated_vote: AggregatedVote domain object

    Returns:
        AggregatedVoteORM model ready for persistence

    Raises:
        ValueError: If aggregated_vote has no llm_judgements
    """
    if not aggregated_vote.llm_judgements:
        raise ValueError("AggregatedVote must have at least one llm_judgement")

    # Extract dataset_sample_id from first judgement (all are for same sample)
    dataset_sample_id = aggregated_vote.llm_judgements[0].llm_prompt.dataset_sample.id

    return AggregatedVoteORM(
        id=aggregated_vote.id,
        dataset_sample_id=dataset_sample_id,
        aggregation_strategy_id=aggregated_vote.aggregation_strategy.id,
        final_label=aggregated_vote.final_label,
        final_confidence=aggregated_vote.final_confidence,
        final_reasoning=aggregated_vote.final_reasoning,
    )


# ============================================================================
# AggregationVote Mappers (junction table linking AggregatedVote → LLMJudgement)
# ============================================================================

def create_aggregation_vote_orm(
    aggregated_vote_id: UUID,
    llm_judgement_id: UUID,
) -> AggregationVoteORM:
    """Create AggregationVoteORM junction record.

    Args:
        aggregated_vote_id: AggregatedVote UUID
        llm_judgement_id: LLMJudgement UUID

    Returns:
        AggregationVoteORM model ready for persistence
    """
    return AggregationVoteORM(
        aggregated_vote_id=aggregated_vote_id,
        llm_judgement_id=llm_judgement_id,
    )


# ============================================================================
# AggregatedDatasetVote Mappers (junction table for many-to-many)
# ============================================================================

def create_aggregated_dataset_vote_orm(
    aggregated_dataset_id: UUID,
    aggregated_vote_id: UUID,
) -> AggregatedDatasetVoteORM:
    """Create AggregatedDatasetVoteORM junction record.

    Args:
        aggregated_dataset_id: AggregatedDataset UUID
        aggregated_vote_id: AggregatedVote UUID

    Returns:
        AggregatedDatasetVoteORM model ready for persistence
    """
    return AggregatedDatasetVoteORM(
        aggregated_dataset_id=aggregated_dataset_id,
        aggregated_vote_id=aggregated_vote_id,
    )
