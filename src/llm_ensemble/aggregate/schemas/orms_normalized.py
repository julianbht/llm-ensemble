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
