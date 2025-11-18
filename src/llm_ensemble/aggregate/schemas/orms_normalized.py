"""
SQLAlchemy ORM models for AGGREGATE CLI.

Pure SQLAlchemy models for database persistence of ensemble aggregation results.
All models use deterministic UUID primary keys computed via uuid_helpers.

FUNCTIONAL DEPENDENCIES & NORMALIZATION:

The schema is in 3NF (Third Normal Form) with the following functional dependencies:

1. AggregationSpecORM:
   - name → id (deterministic UUID)
   - name → description, strategy_module, strategy_class (spec defines implementation)

2. AggregateRunORM:
   - run_name → {id, run_type, aggregation_spec_id, git_sha, git_branch, git_is_dirty, notes, created_at}
   - id → run_name (bidirectional via deterministic UUID)
   - Each run uses exactly ONE aggregation spec (enforced by design)

3. AggregatedScoreORM:
   - id → {aggregate_run_id, final_label, final_confidence, final_reasoning}
   - Aggregation spec is determined by aggregate_run.aggregation_spec_id
   - Result functionally dependent on: (set of llm_calls via join table, aggregation_spec)
   - Individual model votes NOT stored (derivable from llm_call.response.label via join table)

4. AggregatedScoreLLMCallORM:
   - Composite primary key: (aggregated_score_id, llm_call_id)
   - Pure join table (many-to-many between AggregatedScore and LLMCall)
   - No non-key attributes - satisfies BCNF
   - Votes derivable via llm_call.response.label (no denormalization)

DESIGN RATIONALE:

- Enforces one aggregation spec per run (compare specs by running multiple aggregate runs)
- AggregatedScore is the primary output entity (no thin wrapper entity needed)
- Semantic constraints (one score per sample per run) enforced by pipeline, not DB
- AggregatedScoreLLMCall uses composite PK (no surrogate ID or timestamp needed)
- Individual votes not denormalized (derivable from LLMScoreORM.label)
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base, utcnow
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class AggregationSpecORM(Base):
    """Aggregation spec entity - catalog of available ensemble methods.

    Uses deterministic UUID based on spec name.
    One row per spec (majority_vote, weighted_majority, etc.).

    Functional dependencies:
    - name → {description, strategy_module, strategy_class}
    """
    __tablename__ = "aggregation_specs"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = "name"
    __uuid_function__ = "compute_aggregation_spec_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Dynamic adapter specification
    strategy_module = Column(String(512), nullable=False)
    strategy_class = Column(String(255), nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    aggregate_runs = relationship("AggregateRunORM", back_populates="aggregation_spec")


class AggregateRunORM(Base):
    """Aggregate run metadata - execution context for ensemble aggregation.

    Uses deterministic UUID based on run_name.
    Captures which aggregation spec was used and git provenance for reproducibility.

    Functional dependencies:
    - run_name → {id, run_type, aggregation_spec_id, git_sha, git_branch, git_is_dirty, notes}
    - id → run_name (bidirectional via deterministic UUID)
    """
    __tablename__ = "aggregate_runs"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = "run_name"
    __uuid_function__ = "compute_aggregate_run_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False)

    # Aggregation spec used for this run
    aggregation_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregation_specs.id"),
        nullable=False
    )

    # Metadata
    git_sha = Column(String(40), nullable=False)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(String(255), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    aggregation_spec = relationship("AggregationSpecORM", back_populates="aggregate_runs")
    aggregated_scores = relationship("AggregatedScoreORM", back_populates="aggregate_run")


class AggregatedScoreORM(Base):
    """Consensus result from aggregating multiple LLM calls.

    Stores the consensus decision produced by the aggregation strategy.
    The result is functionally dependent on:
    - The set of LLM calls aggregated (via AggregatedScoreLLMCallORM join table)
    - The aggregation spec used (via aggregate_run.aggregation_spec_id)

    Individual model votes are NOT stored here - they're derivable from
    llm_call.score.label via the join table (order-independent).

    Semantic constraint (one score per sample per run) is enforced by the pipeline,
    not by database constraints.
    """
    __tablename__ = "aggregated_scores"
    __table_args__ = {"schema": "aggregate"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    # Reference to aggregate run
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

    Note: Individual model votes are NOT stored here - they are derivable from
    llm_call.score.label. This avoids denormalization and maintains single source of truth.
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
