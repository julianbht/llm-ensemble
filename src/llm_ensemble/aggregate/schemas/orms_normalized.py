"""
SQLAlchemy ORM models for AGGREGATE CLI.

Pure SQLAlchemy models for database persistence of ensemble aggregation results.
All models use deterministic UUID primary keys computed via uuid_helpers.

FUNCTIONAL DEPENDENCIES & NORMALIZATION:

The schema is in 3NF (Third Normal Form) with the following functional dependencies:

1. AggregationStrategyORM:
   - name → id (deterministic UUID)
   - name → description, strategy_module, strategy_class (strategy spec defines implementation)

2. AggregateRunORM:
   - run_name → {id, run_type, strategy_id, git_sha, git_branch, git_is_dirty, notes, created_at}
   - id → run_name (bidirectional via deterministic UUID)
   - Each run uses exactly ONE strategy (enforced by design)

3. AggregatedScoreORM:
   - (judging_sample_id, aggregate_run_id) → {id, final_label, final_confidence, final_reasoning, per_model_votes}
   - One score per (sample, run) pair
   - judging_sample_id is denormalized (derivable from calls) but required for uniqueness constraint
   - Strategy is determined by aggregate_run.strategy_id (no per-score strategy variation)
   - per_model_votes stored as ARRAY (not separate rows) because:
     * Always accessed together (never individually)
     * Order matters (corresponds to judgements list)
     * Variable length per judgement
     * No independent queries on individual votes

4. AggregatedScoreLLMCallORM:
   - Composite primary key: (aggregated_score_id, llm_call_id)
   - Pure join table (many-to-many between AggregatedScore and LLMCall)
   - No non-key attributes - satisfies BCNF

DESIGN RATIONALE:

- Enforces one strategy per run (compare strategies by running multiple aggregate runs)
- AggregatedScore is the primary output entity (no thin wrapper entity needed)
- judging_sample_id stored in AggregatedScore for uniqueness constraint (acceptable denormalization)
- AggregatedScoreLLMCall uses composite PK (no surrogate ID or timestamp needed)
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    ForeignKey,
    UniqueConstraint,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base, utcnow
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class AggregationStrategyORM(Base):
    """Aggregation strategy entity - catalog of available ensemble methods.

    Uses deterministic UUID based on strategy name.
    One row per strategy type (majority_vote, weighted_majority, etc.).

    Functional dependencies:
    - name → {description, strategy_module, strategy_class}
    """
    __tablename__ = "aggregation_strategies"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = "name"
    __uuid_function__ = "compute_aggregation_strategy_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Dynamic adapter specification
    strategy_module = Column(String(512), nullable=False)
    strategy_class = Column(String(255), nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    aggregate_runs = relationship("AggregateRunORM", back_populates="strategy")


class AggregateRunORM(Base):
    """Aggregate run metadata - execution context for ensemble aggregation.
    
    Uses deterministic UUID based on run_name.
    Captures which strategy was used and git provenance for reproducibility.
    
    Functional dependencies:
    - run_name → {id, run_type, strategy_id, git_sha, git_branch, git_is_dirty, notes}
    - id → run_name (bidirectional via deterministic UUID)
    """
    __tablename__ = "aggregate_runs"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = "run_name"
    __uuid_function__ = "compute_aggregate_run_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False)
    
    # Strategy used for this run
    strategy_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregation_strategies.id"),
        nullable=False
    )
    
    # Metadata
    git_sha = Column(String(40), nullable=False)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(String(255), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    strategy = relationship("AggregationStrategyORM", back_populates="aggregate_runs")
    aggregated_scores = relationship("AggregatedScoreORM", back_populates="aggregate_run")


class AggregatedScoreORM(Base):
    """Consensus result from aggregating multiple LLM calls for one sample.

    Stores the consensus decision produced by the aggregation strategy.
    One score per (sample, run) pair. Strategy is determined by aggregate_run.strategy_id.

    Functional dependencies:
    - (judging_sample_id, aggregate_run_id) → {id, final_label, final_confidence, final_reasoning, per_model_votes}

    per_model_votes stored as ARRAY because:
    - Always accessed together (debugging/analysis)
    - Order matters (corresponds to judgements list)
    - No independent queries needed on individual votes

    Note: judging_sample_id is denormalized (derivable from call memberships) but required
    for the uniqueness constraint ensuring one aggregation per sample per run.
    """
    __tablename__ = "aggregated_scores"
    __table_args__ = (
        UniqueConstraint(
            "judging_sample_id",
            "aggregate_run_id",
            name="uq_aggregated_score_sample_run"
        ),
        {"schema": "aggregate"},
    )
    __natural_key__ = ("judging_sample_id", "aggregate_run_id")
    __uuid_function__ = "compute_aggregated_score_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    # References to judging sample (from ingest) and aggregate run
    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id"),
        nullable=False
    )
    aggregate_run_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregate_runs.id"),
        nullable=False
    )

    # Consensus decision outputs
    final_label = Column(
        SQLEnum(RelevanceScore, schema="public"),
        nullable=True
    )
    final_confidence = Column(Float, nullable=True)
    final_reasoning = Column(Text, nullable=False, default="")

    # Per-model votes as ARRAY (ordered, corresponds to judgements list)
    # Format: [0, 1, 2, null, 1] where null = parsing failure
    per_model_votes = Column(ARRAY(Integer), nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    aggregate_run = relationship("AggregateRunORM", back_populates="aggregated_scores")
    call_memberships = relationship("AggregatedScoreLLMCallORM", back_populates="aggregated_score")


class AggregatedScoreLLMCallORM(Base):
    """Join table linking LLM calls to aggregated scores.

    Many-to-many relationship: each aggregated score aggregates multiple LLM calls,
    and each LLM call can potentially be used in multiple aggregation runs.

    Uses composite primary key (aggregated_score_id, llm_call_id).
    Pure join table (BCNF) - no non-key attributes.
    """
    __tablename__ = "aggregated_score_llm_calls"
    __table_args__ = {"schema": "aggregate"}

    aggregated_score_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_scores.id"),
        primary_key=True
    )
    llm_call_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_calls.id"),
        primary_key=True
    )

    # Relationships
    aggregated_score = relationship("AggregatedScoreORM", back_populates="call_memberships")
    # Note: llm_call relationship defined in infer.orms (cross-schema)
