"""
SQLAlchemy ORM models for INFER CLI.
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
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base, utcnow
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class ProviderORM(Base):
    """Provider ORM- LLM provider entity.

    Uses deterministic UUID based on provider name.
    One row per provider (openrouter, ollama, hf).
    """
    __tablename__ = "providers"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = "name"
    __uuid_function__ = "compute_provider_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True),nullable=False,default=utcnow)

    # Relationships
    model_specs = relationship("ModelSpecORM", back_populates="provider")


class PromptTemplateORM(Base):
    """PromptTemplate ORM - prompt template with name.

    Uses deterministic UUID based on template name.
    Each prompt config (e.g., thomas-simple, thomas-advanced) is a distinct entity,
    even if template text is similar. This ensures we know exactly what was used.
    """
    __tablename__ = "prompt_templates"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = "name"
    __uuid_function__ = "compute_prompt_template_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    template_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    infer_runs = relationship("InferRunORM", back_populates="prompt_template")


class ModelSpecORM(Base):
    """ModelSpec ORM specification with inference parameters.
    Captures experimental parameters (model ID, temperature, etc.) that affect LLM behavior.
    """
    __tablename__ = "model_specs"
    __natural_key__ = "name"
    __uuid_function__ = "compute_model_spec_uuid"
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    model_id = Column(String(255), nullable=False)
    provider_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.providers.id"),
        nullable=False
    )
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

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    provider = relationship("ProviderORM", back_populates="model_specs")
    infer_runs = relationship("InferRunORM", back_populates="model_spec")


class InferRunORM(Base):
    __tablename__ = "infer_runs"
    __natural_key__ = "run_name"
    __uuid_function__ = "compute_infer_run_uuid"
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)

    model_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.model_specs.id"),
        nullable=False,
    )
    prompt_template_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.prompt_templates.id"),
        nullable=False,
    )
    parser_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.parser_specs.id"),
        nullable=False,
    )

    input_file = Column(String(1024), nullable=False)
    limit = Column(Integer, nullable=True)
    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships (many runs, one spec/template/model)
    model_spec = relationship("ModelSpecORM", back_populates="infer_runs")
    prompt_template = relationship("PromptTemplateORM", back_populates="infer_runs")
    parser_spec = relationship("ParserSpecORM", back_populates="infer_runs")
    calls = relationship("LLMCallORM", back_populates="infer_run")


class ParserSpecORM(Base):
    __tablename__ = "parser_specs"
    __natural_key__ = ("parser_module", "parser_class", "code_hash")
    __uuid_function__ = "compute_parser_spec_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    code_hash = Column(CHAR(64), nullable=False)
    parser_module = Column(String(512), nullable=False)
    parser_class = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "parser_module",
            "parser_class",
            "code_hash",
            name="uq_parser_spec_identity",
        ),
        {"schema": "infer"},
    )

    infer_runs = relationship("InferRunORM", back_populates="parser_spec")

    # One parser spec can be used by many parsed responses
    responses = relationship("LLMResponseORM", back_populates="parser_spec")


class LLMRequestORM(Base):
    __tablename__ = "llm_requests"
    __natural_key__ = ("prompt", "judging_sample_id")
    __uuid_function__ = "compute_llm_request_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id"),  # Cross-schema FK
        nullable=False,
    )
    prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "prompt",
            "judging_sample_id",
            name="uq_prompt_judging_sample",
        ),
        {"schema": "infer"},
    )

    calls = relationship("LLMCallORM", back_populates="llm_request")


class LLMCallORM(Base):
    __tablename__ = "llm_calls"
    __natural_key__ = ("llm_request_id", "infer_run_id")
    __uuid_function__ = "compute_llm_call_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    llm_request_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_requests.id"),
        nullable=False,
    )
    infer_run_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.infer_runs.id"),
        nullable=False,
    )

    # Each call has at most one (possibly shared) response
    response_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_responses.id"),
        nullable=True,
    )

    latency_ms = Column(Float, nullable=False)
    retries = Column(Integer, nullable=False, default=0)
    cost_estimate_usd = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "llm_request_id",
            "infer_run_id",
            name="uq_call_per_run_and_request",
        ),
        {"schema": "infer"},
    )

    llm_request = relationship("LLMRequestORM", back_populates="calls")
    infer_run = relationship("InferRunORM", back_populates="calls")
    response = relationship("LLMResponseORM", back_populates="calls")


class LLMResponseORM(Base):
    __tablename__ = "llm_responses"
    __natural_key__ = ("parser_spec_id", "raw_response")
    __uuid_function__ = "compute_llm_response_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    parser_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.parser_specs.id"),
        nullable=False,
    )
    raw_response = Column(Text, nullable=False)

    # Derived / parsed info; functionally dependent on (parser_spec_id, raw_response)
    label = Column(SQLEnum(RelevanceScore, schema="public"), nullable=False)
    confidence = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)

    # Parser warnings as JSONB array (data quality issues during parsing)
    # Example: [{"type": "ParserWarning", "code": "field_error", "message": "...", "metadata": {...}}]
    parser_warnings = Column(ARRAY(JSONB), nullable=False, default=[])

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "parser_spec_id",
            "raw_response",
            name="uq_response_parser_raw",
        ),
        {"schema": "infer"},
    )

    # One response can be reused by many calls
    calls = relationship("LLMCallORM", back_populates="response")
    parser_spec = relationship("ParserSpecORM", back_populates="responses")

