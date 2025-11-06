"""SQLAlchemy ORM models for INFER CLI.

Pure SQLAlchemy models (NOT SQLModel) for database persistence.
Separate from Pydantic schemas to maintain clean architecture.

All models use deterministic UUID primary keys computed via uuid_helpers.

Example: LLMJudgementModel (ORM) vs LLMJudgement (Pydantic)
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    ForeignKey,
    UniqueConstraint,
    Index,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base
from llm_ensemble.libs.runtime.run_info import RunType


class WarningStage(str, PyEnum):
    """Warning stage enum for InferWarningModel.

    Differentiates warnings from three pipeline stages.
    """
    PROMPT = "PROMPT"
    PROVIDER = "PROVIDER"
    PARSER = "PARSER"


class ProviderORM(Base):
    """Provider ORM- LLM provider entity.

    Uses deterministic UUID based on provider name.
    One row per provider (openrouter, ollama, hf).
    """
    __tablename__ = "providers"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    model_specs = relationship("ModelSpecORM", back_populates="provider")


class PromptTemplateORM(Base):
    """PromptTemplate ORM - prompt template with name.

    Uses deterministic UUID based on template name.
    Each prompt config (e.g., thomas-simple, thomas-advanced) is a distinct entity,
    even if template text is similar. This ensures we know exactly what was used.
    """
    __tablename__ = "prompt_templates"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    template_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    infer_runs = relationship("InferRunModel", back_populates="prompt_template")


class ModelSpecORM(Base):
    """ModelSpec ORM specification with inference parameters.

    Uses deterministic UUID based on spec name.
    Captures experimental parameters (model ID, temperature, etc.) that affect LLM behavior.
    This is a domain entity, not a config - configs are just YAML plumbing.
    """
    __tablename__ = "model_specs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    model_id = Column(String(255), nullable=False)
    provider_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("providers.id"),
        nullable=False
    )
    provider_module = Column(String(512), nullable=False)
    provider_class = Column(String(255), nullable=False)
    context_window = Column(Integer, nullable=False)

    # Explicit inference parameters for SQL querying
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    top_p = Column(Float, nullable=True)
    frequency_penalty = Column(Float, nullable=True)
    presence_penalty = Column(Float, nullable=True)
    seed = Column(Integer, nullable=True)

    # Additional parameters as JSONB (stop sequences, response_format, etc.)
    additional_params = Column(JSONB, nullable=True)
    capabilities = Column(JSONB, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    provider = relationship("ProviderORM", back_populates="model_specs")
    infer_runs = relationship("InferRunORM", back_populates="model_spec")


class InferRunORM(Base):
    """InferRun ORM - metadata for infer runs.

    Uses deterministic UUID based on run_name.
    Tracks run_type using RunType enum for proper typing and validation.
    References normalized config tables via FKs for SQL querying.
    """
    __tablename__ = "infer_runs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType), nullable=False, default=RunType.TEST)

    # Configuration FKs (normalized for SQL querying)
    model_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_specs.id"),
        nullable=False
    )
    prompt_template_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("prompt_templates.id"),
        nullable=False
    )

    # Input parameters
    input_file = Column(String(1024), nullable=False)
    limit = Column(Integer, nullable=True)

    # Git reproducibility
    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(String(10), nullable=True)

    # Optional notes
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    model_spec = relationship("ModelSpecORM", back_populates="infer_runs")
    prompt_template = relationship("PromptTemplateORM", back_populates="infer_runs")
    llm_judgements = relationship("LLMJudgementORM", back_populates="infer_run")


class LLMJudgementORM(Base):
    """LLMJudgement ORM - denormalized LLM inference results.

    Uses deterministic UUID based on judging_sample_id + infer_run_id.
    Links to JudgingSampleModel (from INGEST) and InferRunModel via foreign keys.

    Denormalizes request/response/score fields for simpler queries:
    - Request fields: prompt
    - Response fields: raw_response, latency_ms, retries, cost_estimate_usd
    - Score fields: label, confidence, rationale

    Stores run_name (denormalized) for easy querying without joins.
    """
    __tablename__ = "llm_judgements"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("judging_samples.id"),
        nullable=False
    )
    infer_run_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer_runs.id"),
        nullable=False
    )

    # Request fields (from LLMRequest)
    prompt = Column(Text, nullable=False)

    # Response fields (from LLMResponse)
    raw_response = Column(Text, nullable=False)
    latency_ms = Column(Float, nullable=False)
    retries = Column(Integer, nullable=False)
    cost_estimate_usd = Column(Float, nullable=True)

    # Score fields (from LLMScore - nullable if parse failed)
    label = Column(Integer, nullable=True)  # 0/1/2/3 or NULL
    confidence = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    infer_run = relationship("InferRunORM", back_populates="llm_judgements")
    warnings = relationship("InferWarningORM", back_populates="judgement")

    __table_args__ = (
        UniqueConstraint(
            "judging_sample_id",
            "infer_run_id",
            name="uq_judgement_sample_run"
        ),
    )


class InferWarningORM(Base):
    """InferWarning ORM - warnings from all pipeline stages.

    Uses deterministic UUID based on judgement_id + stage + code + message_hash.
    Polymorphic table storing warnings from three stages:
    - PROMPT: PromptBuilder warnings (rendering errors, validation)
    - PROVIDER: LLMProvider warnings (API errors, retries)
    - PARSER: ResponseParser warnings (parse errors, field errors)

    Stores metadata as JSONB for flexible analytics.
    """
    __tablename__ = "infer_warnings"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    judgement_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("llm_judgements.id"),
        nullable=False
    )
    stage = Column(SQLEnum(WarningStage), nullable=False)
    code = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    judgement = relationship("LLMJudgementORM", back_populates="warnings")

    __table_args__ = (
        Index("idx_warning_judgement_stage", "judgement_id", "stage"),
    )
