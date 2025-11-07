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
    ForeignKeyConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship

from llm_ensemble.libs.db import Base
from llm_ensemble.libs.runtime.run_info import RunType


class WarningStage(str, PyEnum):
    """Warning stage enum for InferWarningORM."""
    PROMPT = "PROMPT"
    PROVIDER = "PROVIDER"
    PARSER = "PARSER"


class ProviderORM(Base):
    """LLM provider entity (openrouter, ollama, hf, etc.)."""

    __tablename__ = "providers"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    model_specs = relationship("ModelSpecORM", back_populates="provider")


class PromptTemplateORM(Base):
    """Prompt template with a stable name."""

    __tablename__ = "prompt_templates"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    template_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    infer_runs = relationship("InferRunORM", back_populates="prompt_template")


class ModelSpecORM(Base):
    """Model specification with provider + inference parameters."""

    __tablename__ = "model_specs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)

    model_id = Column(String(255), nullable=False)

    provider_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("providers.id"),
        nullable=False,
    )

    provider_module = Column(String(512), nullable=False)
    provider_class = Column(String(255), nullable=False)
    context_window = Column(Integer, nullable=False)

    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    top_p = Column(Float, nullable=True)
    frequency_penalty = Column(Float, nullable=True)
    presence_penalty = Column(Float, nullable=True)
    seed = Column(Integer, nullable=True)

    additional_params = Column(JSONB, nullable=True)
    capabilities = Column(JSONB, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    provider = relationship("ProviderORM", back_populates="model_specs")
    infer_runs = relationship("InferRunORM", back_populates="model_spec")


class ParserSpecORM(Base):
    """Identifies a concrete response parser implementation."""

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


class InferRunORM(Base):
    """One concrete evaluation / run configuration."""

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

    parsed_results = relationship("ParsedResultORM", back_populates="infer_run")
    llm_judgements = relationship("LLMJudgementORM", back_populates="infer_run")


class ParsedResultORM(Base):
    """
    ParsedResult belongs to a specific InferRun.

    Semantics:
        For a given run (and thus a fixed parser_spec), this is how a raw_response
        was parsed (label, confidence, rationale).

    Invariant:
        - infer_run_id -> infer_runs.id
        - (infer_run_id, raw_response) is unique for de-dup within a run.
    """

    __tablename__ = "parsed_results"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    infer_run_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer_runs.id"),
        nullable=False,
    )

    raw_response = Column(Text, nullable=False)

    label = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        # Within a given run, same raw_response maps to one ParsedResult
        UniqueConstraint(
            "infer_run_id",
            "raw_response",
            name="uq_parsed_result_run_response",
        ),
        # For composite FK from LLMJudgement to (id, infer_run_id)
        UniqueConstraint(
            "id",
            "infer_run_id",
            name="uq_parsed_result_id_run",
        ),
    )

    # Relationships
    infer_run = relationship("InferRunORM", back_populates="parsed_results")
    llm_judgements = relationship("LLMJudgementORM", back_populates="parsed_result")


class LLMJudgementORM(Base):
    """
    Normalized judgement per (judging_sample, infer_run).

    - Links a judging_sample to a specific infer_run.
    - Optionally references a ParsedResult from THAT SAME run.
    - Stores request/metric info.

    DB-enforced invariant (no triggers):
        If parsed_result_id is non-null:
            parsed_results.infer_run_id == llm_judgements.infer_run_id
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

    # Optional link to ParsedResult; must belong to the same run (see FK below)
    parsed_result_id = Column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )

    # Request data
    prompt = Column(Text, nullable=False)

    # Metrics
    latency_ms = Column(Float, nullable=False)
    cost_estimate_usd = Column(Float, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        # One judgement per (sample, run)
        UniqueConstraint(
            "judging_sample_id",
            "infer_run_id",
            name="uq_judgement_sample_run",
        ),

        # Enforce: if parsed_result_id set, it must be from the same infer_run
        ForeignKeyConstraint(
            ["parsed_result_id", "infer_run_id"],
            ["parsed_results.id", "parsed_results.infer_run_id"],
            name="fk_judgement_parsed_result_same_run",
            ondelete="SET NULL",
        ),
    )

    # Relationships
    infer_run = relationship("InferRunORM", back_populates="llm_judgements")

    parsed_result = relationship(
        "ParsedResultORM",
        back_populates="llm_judgements",
        primaryjoin=(
            "and_(LLMJudgementORM.parsed_result_id==ParsedResultORM.id, "
            "LLMJudgementORM.infer_run_id==ParsedResultORM.infer_run_id)"
        ),
    )

    warnings = relationship(
        "InferWarningORM",
        back_populates="judgement",
        cascade="all, delete-orphan",
    )


class InferWarningORM(Base):
    """Polymorphic warnings for PROMPT / PROVIDER / PARSER stages."""

    __tablename__ = "infer_warnings"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    judgement_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("llm_judgements.id"),
        nullable=False,
    )

    stage = Column(SQLEnum(WarningStage), nullable=False)
    code = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    judgement = relationship("LLMJudgementORM", back_populates="warnings")
