"""
SQLAlchemy ORM models for AGGREGATE CLI.
Pure SQLAlchemy models for database persistence.
All models use deterministic UUID primary keys computed via uuid_helpers.
"""

from __future__ import annotations

from sqlalchemy import (
    CHAR,
    Column,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    UniqueConstraint,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db.base import Base
from llm_ensemble.libs.db.utcnow import utcnow
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class AggregationStrategyORM(Base):
    """Minimal entity tracking which aggregation strategy was used.

    Just id + name - no wiring details (module/class paths).
    Name comes from adapter's strategy_name property (e.g., 'majority_vote').
    Referenced via aggregate_run_configs.
    """
    __tablename__ = "aggregation_strategies"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = "name"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True, comment="Natural key from adapter.strategy_name (e.g., 'majority_vote')")

    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    aggregate_run_configs = relationship("AggregateRunConfigORM", back_populates="aggregation_strategy")


class AggregateRunConfigORM(Base):
    """Configuration for an aggregate run."""
    __tablename__ = "aggregate_run_configs"
    __table_args__ = (
        UniqueConstraint(
            "aggregation_strategy_id",
            "io_config_name",
            "input_run_names_hash",
            name="uq_aggregate_run_config",
        ),
        {"schema": "aggregate"},
    )
    __natural_key__ = ("aggregation_strategy_id", "io_config_name", "input_run_names_hash")

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    aggregation_strategy_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregation_strategies.id"),
        nullable=False,
    )
    io_config_name = Column(String(255), nullable=False)
    input_run_names = Column(JSONB, nullable=False, comment="List of infer run identifiers")
    input_run_names_hash = Column(CHAR(64), nullable=False, comment="SHA256 of sorted input_run_names for uniqueness")
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    aggregation_strategy = relationship("AggregationStrategyORM", back_populates="aggregate_run_configs")
    aggregate_runs = relationship("AggregateRunORM", back_populates="aggregate_run_config")


class AggregateRunORM(Base):
    """Complete record of an aggregate execution.

    Connects configuration (input) to dataset (output) with execution metadata.
    """
    __tablename__ = "aggregate_runs"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = ("run_name",)

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)

    # What was intended (configuration)
    aggregate_run_config_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregate_run_configs.id"),
        nullable=False,
        comment="Configuration used for this run"
    )

    # What was produced (output)
    aggregated_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_datasets.id"),
        nullable=False,
        comment="Dataset produced by this run"
    )

    # Timing
    start_time = Column(DateTime, nullable=False, comment="When the run started")
    end_time = Column(DateTime, nullable=False, comment="When the run completed")

    # Git metadata for reproducibility
    git_sha = Column(String(40), nullable=False)
    git_branch = Column(String(255), nullable=False)
    git_is_dirty = Column(String(10), nullable=False)

    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    aggregate_run_config = relationship("AggregateRunConfigORM", back_populates="aggregate_runs")
    aggregated_dataset = relationship("AggregatedDatasetORM", back_populates="aggregate_runs")


class AggregatedDatasetORM(Base):
    __tablename__ = "aggregated_datasets"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = ("fingerprint",)

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    fingerprint = Column(
        CHAR(64),
        nullable=False,
        unique=True,
        comment="SHA256 of sorted dataset_sample IDs (identifies which samples were aggregated)"
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    aggregate_runs = relationship("AggregateRunORM", back_populates="aggregated_dataset")
    # Many-to-many with AggregatedVoteORM via join table
    aggregated_votes = relationship(
        "AggregatedVoteORM",
        secondary="aggregate.aggregated_dataset_votes",
        back_populates="aggregated_datasets"
    )


class AggregatedVoteORM(Base):
    __tablename__ = "aggregated_votes"
    __table_args__ = {"schema": "aggregate"}
    __natural_key__ = ("judgement_fingerprint", "final_label", "final_confidence", "final_reasoning")

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    judgement_fingerprint = Column(
        CHAR(64),
        nullable=False,
        comment="SHA256 hash of sorted judgement IDs that were aggregated"
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
            "judgement_fingerprint",
            "final_label",
            "final_confidence",
            "final_reasoning",
            name="uq_aggregated_vote_identity",
        ),
        {"schema": "aggregate"},
    )

    # Relationships
    aggregation_votes = relationship("AggregationVoteORM", back_populates="aggregated_vote")
    # Many-to-many with AggregatedDatasetORM via join table
    aggregated_datasets = relationship(
        "AggregatedDatasetORM",
        secondary="aggregate.aggregated_dataset_votes",
        back_populates="aggregated_votes"
    )


# Join table for many-to-many relationship between AggregatedDataset and AggregatedVote
class AggregatedDatasetVoteORM(Base):
    __tablename__ = "aggregated_dataset_votes"
    __table_args__ = {"schema": "aggregate"}

    aggregated_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_datasets.id", ondelete="CASCADE"),
        primary_key=True,
    )
    aggregated_vote_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_votes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)


class AggregationVoteORM(Base):
    __tablename__ = "aggregation_votes"
    __table_args__ = {"schema": "aggregate"}

    aggregated_vote_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("aggregate.aggregated_votes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    llm_judgement_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_judgements.id"),
        primary_key=True,
    )

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    aggregated_vote = relationship("AggregatedVoteORM", back_populates="aggregation_votes")
    llm_judgement = relationship(
        "LLMJudgementORM",
        foreign_keys=[llm_judgement_id]
    )
