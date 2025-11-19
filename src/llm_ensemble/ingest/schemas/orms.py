"""SQLAlchemy ORM models for INGEST CLI.

Pure SQLAlchemy models (NOT SQLModel) for database persistence.
Separate from Pydantic schemas to maintain clean architecture.

All models use deterministic UUID primary keys computed via uuid_helpers.

Naming convention: ORM models use "Model" suffix to distinguish from Pydantic schemas.
Example: JudgingSampleModel (ORM) vs JudgingSample (Pydantic)
"""

from __future__ import annotations
from datetime import datetime

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


class DatasetORM(Base):
    """Dataset ORM model - normalized dataset metadata.

    Each dataset represents a distinct IR dataset (e.g., 'msmarco', 'trec-covid').
    Uses deterministic UUID based on dataset name.
    """
    __tablename__ = "datasets"
    __table_args__ = {"schema": "ingest"}
    __natural_key__ = ("name",)
    __uuid_function__ = "compute_dataset_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    
    # Relationships
    queries = relationship("QueryORM", back_populates="dataset")
    documents = relationship("DocumentORM", back_populates="dataset")


class QueryORM(Base):
    """Query ORM model - search queries from IR datasets.

    Uses deterministic UUID based on dataset + external_id.
    """
    __tablename__ = "queries"
    __natural_key__ = ("dataset_id", "external_id")
    __uuid_function__ = "compute_query_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey("ingest.datasets.id"), nullable=False)
    external_id = Column(String(255), nullable=False)
    query_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    
    # Relationships
    dataset = relationship("DatasetORM", back_populates="queries")
    judging_samples = relationship("JudgingSampleORM", back_populates="query")
    
    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id", name="uq_query_dataset_external_id"),
        {"schema": "ingest"},
    )


class DocumentORM(Base):
    """Document ORM model - documents from IR datasets.

    Uses deterministic UUID based on dataset + external_id.
    """
    __tablename__ = "documents"
    __natural_key__ = ("dataset_id", "external_id")
    __uuid_function__ = "compute_document_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey("ingest.datasets.id"), nullable=False)
    external_id = Column(String(255), nullable=False)
    doc_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    
    # Relationships
    dataset = relationship("DatasetORM", back_populates="documents")
    judging_samples = relationship("JudgingSampleORM", back_populates="document")

    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id", name="uq_document_dataset_external_id"),
        {"schema": "ingest"},
    )


class NormalizedDatasetORM(Base):
    """NormalizedDataset ORM - internal dataset with deterministic fingerprint.

    Represents a specific collection of judging samples. Multiple ingest runs
    can produce the same NormalizedDataset (same fingerprint), enabling idempotency.

    Uses deterministic UUID based on fingerprint (SHA256 of sorted sample IDs).
    """
    __tablename__ = "normalized_datasets"
    __table_args__ = {"schema": "ingest"}
    __natural_key__ = ("fingerprint",)
    __uuid_function__ = "compute_normalized_dataset_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    fingerprint = Column(CHAR(64), nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judging_samples = relationship(
        "JudgingSampleORM",
        secondary="ingest.normalized_dataset_judging_samples",
        back_populates="normalized_datasets",
        order_by="NormalizedDatasetJudgingSampleORM.sequence_number"
    )
    ingest_runs = relationship("IngestRunORM", back_populates="normalized_dataset")


class NormalizedDatasetJudgingSampleORM(Base):
    """Junction table linking NormalizedDataset to JudgingSample with sequence.

    Preserves deterministic ordering of samples via sequence_number.
    This enables reproducible slicing (start_sample/end_sample) in future.
    """
    __tablename__ = "normalized_dataset_judging_samples"
    __table_args__ = {"schema": "ingest"}

    normalized_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.normalized_datasets.id", ondelete="CASCADE"),
        primary_key=True,
    )
    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class IngestRunORM(Base):
    """IngestRun ORM model - metadata for ingest runs.

    Uses deterministic UUID based on run_name.
    Tracks run_type using RunType enum for proper typing and validation.

    Each ingest run produces exactly one NormalizedDataset.
    Multiple runs can produce the same NormalizedDataset (idempotency).
    """
    __tablename__ = "ingest_runs"
    __table_args__ = {"schema": "ingest"}
    __natural_key__ = ("run_name",)
    __uuid_function__ = "compute_ingest_run_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)
    normalized_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.normalized_datasets.id"),
        nullable=False
    )
    io_config_name = Column(String(255), nullable=False)
    input_path = Column(String(1024), nullable=False)
    limit = Column(Integer, nullable=True)
    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(String(10), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    normalized_dataset = relationship("NormalizedDatasetORM", back_populates="ingest_runs")


class JudgingSampleORM(Base):
    """JudgingSample ORM - query-document pairs with gold relevance scores.

    Uses deterministic UUID based on query_id + document_id.
    Links to Query and Document via foreign keys.
    Uses RelevanceScore enum for proper typing and validation.

    Relationship to NormalizedDatasets is Many-to-Many via junction table,
    enabling the same sample to be part of multiple normalized datasets.
    """
    __tablename__ = "judging_samples"
    __natural_key__ = ("query_id", "document_id")
    __uuid_function__ = "compute_judging_sample_uuid"

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
        secondary="ingest.normalized_dataset_judging_samples",
        back_populates="judging_samples"
    )

    __table_args__ = (
        UniqueConstraint("query_id", "document_id", name="uq_query_doc"),
        {"schema": "ingest"},
    )
