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
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.schemas import RelevanceScore


class DatasetORM(Base):
    """Dataset ORM model - normalized dataset metadata.
    
    Each dataset represents a distinct IR dataset (e.g., 'msmarco', 'trec-covid').
    Uses deterministic UUID based on dataset name.
    """
    __tablename__ = "datasets"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    queries = relationship("QueryModel", back_populates="dataset")
    documents = relationship("DocumentModel", back_populates="dataset")


class QueryORM(Base):
    """Query ORM model - search queries from IR datasets.
    
    Uses deterministic UUID based on dataset + external_id.
    """
    __tablename__ = "queries"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    external_id = Column(String(255), nullable=False)
    query_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("DatasetModel", back_populates="queries")
    judging_samples = relationship("JudgingSampleModel", back_populates="query")
    
    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id", name="uq_query_dataset_external_id"),
    )


class DocumentORM(Base):
    """Document ORM model - documents from IR datasets.
    
    Uses deterministic UUID based on dataset + external_id.
    """
    __tablename__ = "documents"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    external_id = Column(String(255), nullable=False)
    doc_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("DatasetModel", back_populates="documents")
    judging_samples = relationship("JudgingSampleModel", back_populates="document")
    
    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id", name="uq_document_dataset_external_id"),
    )


class IngestRunORM(Base):
    """IngestRun ORM model - metadata for ingest runs.
    
    Uses deterministic UUID based on run_name.
    Tracks run_type using RunType enum for proper typing and validation.
    """
    __tablename__ = "ingest_runs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType), nullable=False, default=RunType.TEST)
    io_config_name = Column(String(255), nullable=False)
    input_path = Column(String(1024), nullable=False)
    limit = Column(Integer, nullable=True)
    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(String(10), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    judging_samples = relationship("JudgingSampleModel", back_populates="ingest_run")


class JudgingSampleORM(Base):
    """JudgingSample ORM model - query-document pairs with gold relevance scores.

    Uses deterministic UUID based on dataset + query_external_id + doc_external_id.
    Links to Query, Document, Dataset, and IngestRun via foreign keys.
    Uses RelevanceScore enum for proper typing and validation.

    Stores both ingest_run_id (UUID FK) and run_name (denormalized string) for easier querying.
    """
    __tablename__ = "judging_samples"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    query_id = Column(PG_UUID(as_uuid=True), ForeignKey("queries.id"), nullable=False)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    ingest_run_id = Column(PG_UUID(as_uuid=True), ForeignKey("ingest_runs.id"), nullable=False)
    run_name = Column(String(255), nullable=False)  # Denormalized for easy querying
    gold_score = Column(SQLEnum(RelevanceScore), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    query = relationship("QueryModel", back_populates="judging_samples")
    document = relationship("DocumentModel", back_populates="judging_samples")
    ingest_run = relationship("IngestRunModel", back_populates="judging_samples")
    
    __table_args__ = (
        UniqueConstraint("query_id", "document_id", name="uq_query_doc"),
    )
