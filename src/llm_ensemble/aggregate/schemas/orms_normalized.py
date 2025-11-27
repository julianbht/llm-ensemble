"""
SQLAlchemy ORM models for AGGREGATE CLI.
Pure SQLAlchemy models for database persistence.
All models use deterministic UUID primary keys computed via uuid_helpers.
"""

from __future__ import annotations

from sqlalchemy import (
    CHAR,
    Boolean,
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
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
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
    aggregated_votes = relationship("AggregatedVoteORM", back_populates="aggregation_spec")


class AggregateRunORM(Base):
    __tablename__ = "aggregate_runs"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = "run_name"
    __uuid_function__ = "compute_aggregate_run_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)

    # Config names snapshot for easy viewing
    config_names = Column(
        JSONB,
        nullable=False,
        comment="Config names used: {aggregation_spec, io_config}"
    )

    # ACTUAL RESULT: What was actually aggregated (set in close())
    aggregated_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_datasets.id"),
        nullable=True,
        comment="What aggregations were actually produced (NULL = run incomplete/failed)"
    )

    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    aggregated_dataset = relationship("AggregatedDatasetORM", back_populates="aggregate_runs")


class AggregatedDatasetORM(Base):
    __tablename__ = "aggregated_datasets"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = ("fingerprint",)
    __uuid_function__ = "compute_aggregated_dataset_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    fingerprint = Column(
        CHAR(64),
        nullable=False,
        unique=True,
        comment="SHA256 of sorted dataset_vote IDs (identifies which samples were aggregated)"
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    dataset_votes = relationship("DatasetVoteORM", back_populates="aggregated_dataset")
    aggregate_runs = relationship("AggregateRunORM", back_populates="aggregated_dataset")


class DatasetVoteORM(Base):
    __tablename__ = "dataset_votes"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = ("aggregated_dataset_id", "sequence_number")
    __uuid_function__ = "compute_dataset_vote_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    aggregated_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "aggregated_dataset_id",
            "sequence_number",
            name="uq_dataset_vote_position",
        ),
        {"schema": "aggregate"},
    )

    # Relationships
    aggregated_dataset = relationship("AggregatedDatasetORM", back_populates="dataset_votes")
    aggregated_votes = relationship("AggregatedVoteORM", back_populates="dataset_vote")


class AggregatedVoteORM(Base):
    __tablename__ = "aggregated_votes"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = ("dataset_vote_id", "aggregation_spec_id")
    __uuid_function__ = "compute_aggregated_vote_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    dataset_vote_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.dataset_votes.id", ondelete="CASCADE"),
        nullable=False,
    )
    aggregation_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregation_specs.id"),
        nullable=False,
    )

    # Aggregated results
    final_label = Column(
        SQLEnum(RelevanceScore, schema="public"),
        nullable=True,
        comment="Consensus relevance label from aggregation strategy"
    )
    final_confidence = Column(
        Float,
        nullable=True,
        comment="Confidence in aggregated decision [0-1]"
    )
    final_reasoning = Column(
        Text,
        nullable=True,
        comment="Explanation of how consensus was reached"
    )

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "dataset_vote_id",
            "aggregation_spec_id",
            name="uq_aggregated_vote_identity",
        ),
        {"schema": "aggregate"},
    )

    # Relationships
    dataset_vote = relationship("DatasetVoteORM", back_populates="aggregated_votes")
    aggregation_spec = relationship("AggregationSpecORM", back_populates="aggregated_votes")
    aggregation_votes = relationship("AggregationVoteORM", back_populates="aggregated_vote")


class AggregationVoteORM(Base):
    __tablename__ = "aggregation_votes"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = ("aggregated_vote_id", "llm_judgement_id")
    __uuid_function__ = "compute_aggregation_vote_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    aggregated_vote_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_votes.id", ondelete="CASCADE"),
        nullable=False,
    )
    llm_judgement_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_judgements.id"),
        nullable=False,
    )

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "aggregated_vote_id",
            "llm_judgement_id",
            name="uq_aggregation_vote_identity",
        ),
        {"schema": "aggregate"},
    )

    # Relationships
    aggregated_vote = relationship("AggregatedVoteORM", back_populates="aggregation_votes")
    # Note: llm_judgement relationship defined in infer schema (cross-schema)
