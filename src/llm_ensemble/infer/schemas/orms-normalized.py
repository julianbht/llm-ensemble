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
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base
from llm_ensemble.libs.runtime.run_info import RunType


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
    __tablename__ = "infer_runs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType), nullable=False, default=RunType.TEST)

    model_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_specs.id"),
        nullable=False,
    )
    prompt_template_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("prompt_templates.id"),
        nullable=False,
    )

    parser_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("parser_specs.id"),
        nullable=False,
    )

    input_file = Column(String(1024), nullable=False)
    limit = Column(Integer, nullable=True)

    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(String(10), nullable=True)

    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    model_spec = relationship("ModelSpecORM", back_populates="infer_runs")
    prompt_template = relationship("PromptTemplateORM", back_populates="infer_runs")
    parser_spec = relationship("ParserSpecORM", back_populates="infer_runs")
    llm_judgements = relationship("LLMJudgementORM", back_populates="infer_run")


class ParserSpecORM(Base):
    """ParserSpec ORM - identifies a concrete response parser implementation.

    Deterministic UUID based on (parser_module, parser_class).
    Matches PromptConfig.parser_module / parser_class.
    """
    __tablename__ = "parser_specs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    parser_module = Column(String(512), nullable=False)
    parser_class = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "parser_module",
            "parser_class",
            name="uq_parser_spec_module_class",
        ),
    )

    # Relationships
    infer_runs = relationship("InferRunORM", back_populates="parser_spec")
    parsed_results = relationship("ParsedResultORM", back_populates="parser_spec")


class ParsedResultORM(Base):
    """ParsedResult ORM - normalized deterministic parser output.

    Captures:
        (parser_spec_id, raw_response) -> (label, confidence, rationale)

    This is in BCNF because (parser_spec_id, raw_response) is a key.
    """
    __tablename__ = "parsed_results"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    parser_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("parser_specs.id"),
        nullable=False,
    )
    raw_response = Column(Text, nullable=False)

    label = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "parser_spec_id",
            "raw_response",
            name="uq_parsed_result_parser_response",
        ),
    )

    parser_spec = relationship("ParserSpecORM", back_populates="parsed_results")
    llm_judgements = relationship("LLMJudgementORM", back_populates="parsed_result")



class LLMJudgementORM(Base):
    """LLMJudgement ORM - normalized judgement per (sample, run).

    One row per (judging_sample_id, infer_run_id).

    - Stores linkage sample <-> run
    - Stores prompt + runtime metrics
    - References ParsedResult for the actual parsed label/score.
    """
    __tablename__ = "llm_judgements"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("judging_samples.id"),
        nullable=False,
    )

    infer_run_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer_runs.id"),
        nullable=False,
    )

    # Which parsed result this judgement used (may be NULL if parsing failed)
    parsed_result_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("parsed_results.id"),
        nullable=True,
    )

    # Request data
    prompt = Column(Text, nullable=False)

    # Metrics
    latency_ms = Column(Float, nullable=False)
    cost_estimate_usd = Column(Float, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "judging_sample_id",
            "infer_run_id",
            name="uq_judgement_sample_run",
        ),
    )

    infer_run = relationship("InferRunORM", back_populates="llm_judgements")
    parsed_result = relationship("ParsedResultORM", back_populates="llm_judgements")
