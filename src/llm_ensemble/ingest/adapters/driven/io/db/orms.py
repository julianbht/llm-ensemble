"""SQLAlchemy ORM models for INGEST CLI.

Naming convention: ORM models use "Model" suffix to distinguish from Pydantic schemas.
Example: JudgingSampleModel (ORM) vs JudgingSample (Pydantic)

Design:
- Queries and Documents are global entities identified by content hash
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    UniqueConstraint,
    Enum as SQLEnum,
    CHAR,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base
from llm_ensemble.libs.db.utcnow import utcnow
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.schemas import RelevanceScore


class QueryORM(Base):
    __tablename__ = "queries"
    __natural_key__ = ("content_hash",)

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    content_hash = Column(CHAR(64), nullable=False, unique=True)
    query_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judging_samples = relationship("JudgingSampleORM", back_populates="query")

    __table_args__ = {"schema": "ingest"}


class DocumentORM(Base):
    __tablename__ = "documents"
    __natural_key__ = ("content_hash",)

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    content_hash = Column(CHAR(64), nullable=False, unique=True)
    doc_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judging_samples = relationship("JudgingSampleORM", back_populates="document")

    __table_args__ = {"schema": "ingest"}


class NormalizedDatasetORM(Base):
    __tablename__ = "normalized_datasets"
    __table_args__ = {"schema": "ingest"}
    __natural_key__ = ("fingerprint",)

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    fingerprint = Column(CHAR(64), nullable=False, unique=True)
    external_dataset_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judging_samples = relationship(
        "JudgingSampleORM",
        secondary="ingest.dataset_sample",
        back_populates="normalized_datasets",
        order_by="DatasetSampleORM.sequence_number"
    )
    ingest_runs = relationship("IngestRunORM", back_populates="normalized_dataset")


class DatasetSampleORM(Base):
    __tablename__ = "dataset_sample"
    __natural_key__ = ("normalized_dataset_id", "judging_sample_id")

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    normalized_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.normalized_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships for eager loading
    judging_sample = relationship("JudgingSampleORM", lazy="select")

    __table_args__ = (
        UniqueConstraint("normalized_dataset_id", "judging_sample_id", name="uq_dataset_sample"),
        {"schema": "ingest"},
    )


class IngestRunConfigORM(Base):
    """Configuration for an ingest run."""
    __tablename__ = "ingest_run_configs"
    __table_args__ = (
        UniqueConstraint(
            "io_config_name",
            "input_path",
            "limit",
            name="uq_ingest_run_config",
        ),
        {"schema": "ingest"},
    )
    __natural_key__ = ("io_config_name", "input_path", "limit")

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    io_config_name = Column(String(255), nullable=False)
    input_path = Column(String(1024), nullable=False)
    limit = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    ingest_runs = relationship("IngestRunORM", back_populates="ingest_run_config")


class IngestRunORM(Base):
    """Complete record of an ingest execution.

    Connects configuration (input) to dataset (output) with execution metadata.
    """
    __tablename__ = "ingest_runs"
    __table_args__ = {"schema": "ingest"}
    __natural_key__ = ("run_name",)

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)

    # What was intended (configuration)
    ingest_run_config_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.ingest_run_configs.id"),
        nullable=False,
        comment="Configuration used for this run"
    )

    # What was produced (output)
    normalized_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.normalized_datasets.id"),
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
    ingest_run_config = relationship("IngestRunConfigORM", back_populates="ingest_runs")
    normalized_dataset = relationship("NormalizedDatasetORM", back_populates="ingest_runs")


class JudgingSampleORM(Base):
    __tablename__ = "judging_samples"
    __natural_key__ = ("query_id", "document_id")

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    query_id = Column(PG_UUID(as_uuid=True), ForeignKey("ingest.queries.id"), nullable=False)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("ingest.documents.id"), nullable=False)
    gold_score = Column(SQLEnum(RelevanceScore, schema="public"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    query = relationship("QueryORM", back_populates="judging_samples")
    document = relationship("DocumentORM", back_populates="judging_samples")
    normalized_datasets = relationship(
        "NormalizedDatasetORM",
        secondary="ingest.dataset_sample",
        back_populates="judging_samples"
    )

    __table_args__ = (
        UniqueConstraint("query_id", "document_id", name="uq_query_doc"),
        {"schema": "ingest"},
    )
