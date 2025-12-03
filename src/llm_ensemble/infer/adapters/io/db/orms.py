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
    __tablename__ = "providers"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = "name"
    __uuid_function__ = "compute_provider_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    judged_datasets = relationship("JudgedDatasetORM", back_populates="provider")


class ModelConfigORM(Base):
    """Complete model configuration used for an inference run.

    Pure configuration data - model identity, capabilities, and inference parameters.
    No provider reference - provider is a runtime fact on JudgedDataset.
    No separate Model entity - this is the complete configuration snapshot.
    """
    __tablename__ = "model_configs"
    __natural_key__ = "name"
    __uuid_function__ = "compute_model_config_uuid"
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)

    # Model identity
    model_id = Column(String(255), nullable=False, comment="Model identifier (e.g., 'gpt-4', 'llama-3-70b')")

    # Model capabilities (from model_specs in config)
    context_window = Column(Integer, nullable=False)
    capabilities = Column(JSONB, nullable=True, comment="Model capabilities (e.g., multilingual, function_calling)")

    # Inference parameters (from model_specs in config)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    top_p = Column(Float, nullable=True)
    frequency_penalty = Column(Float, nullable=True)
    presence_penalty = Column(Float, nullable=True)
    seed = Column(Integer, nullable=True)

    # Additional parameters as JSONB (stop sequences, response_format, etc.)
    additional_params = Column(JSONB, nullable=True)

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judged_datasets = relationship("JudgedDatasetORM", back_populates="model_config")


class PromptTemplateORM(Base):
    __tablename__ = "prompt_templates"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = "name"
    __uuid_function__ = "compute_prompt_template_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    template_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    prompt_texts = relationship("LLMPromptTextORM", back_populates="prompt_template")


class ParserORM(Base):
    """Minimal entity tracking which response parser was used.

    Just id + name - no wiring details (module/class paths).
    Name comes from registry (e.g., 'thomas-simple').
    """
    __tablename__ = "parser"
    __natural_key__ = "name"
    __uuid_function__ = "compute_parser_spec_uuid_from_name"
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="Natural key from registry (e.g., 'thomas-simple')"
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    scores = relationship("LLMScoreORM", back_populates="parser_spec")


class JudgedDatasetORM(Base):
    """Dataset of LLM judgements produced by an inference run.

    Captures both what was configured (model_config) and where it ran (provider).
    """
    __tablename__ = "judged_datasets"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = None  # 1:1 with InferRun, uses same ID
    __uuid_function__ = None  # ID comes from InferRun

    id = Column(PG_UUID(as_uuid=True), primary_key=True, comment="Same as InferRun.id (1:1 relationship)")

    model_config_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.model_configs.id"),
        nullable=False,
        comment="Which model configuration was used for all judgements in this dataset"
    )

    provider_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.providers.id"),
        nullable=False,
        comment="Which provider/service was used to run inference (runtime fact)"
    )

    sample_fingerprint = Column(
        CHAR(64),
        nullable=True,
        comment="SHA256 of sorted dataset_sample IDs (identifies which samples were judged, for aggregation)"
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    model_config = relationship("ModelConfigORM", back_populates="judged_datasets")
    provider = relationship("ProviderORM", back_populates="judged_datasets")
    llm_judgements = relationship("LLMJudgementORM", back_populates="judged_dataset")
    infer_run = relationship("InferRunORM", back_populates="judged_dataset", uselist=False)


class InferRunORM(Base):
    __tablename__ = "infer_runs"
    __natural_key__ = "run_name"
    __uuid_function__ = "compute_infer_run_uuid"
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)

    # Config names snapshot for easy viewing
    config_names = Column(
        JSONB,
        nullable=False,
        comment="Config names used: {model_config, prompt_template, parser_spec}"
    )

    # INTENT: What range of NormalizedDataset to process (set in open())
    start_idx = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Start index into NormalizedDataset.samples (0-indexed, inclusive)"
    )
    end_idx = Column(
        Integer,
        nullable=False,
        comment="End index into NormalizedDataset.samples (exclusive)"
    )

    # ACTUAL RESULT: What was actually judged (set in close())
    judged_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.judged_datasets.id"),
        nullable=True,
        comment="What judgements were actually produced (NULL = run incomplete/failed)"
    )

    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judged_dataset = relationship("JudgedDatasetORM", back_populates="infer_run", uselist=False)


class LLMPromptTextORM(Base):
    __tablename__ = "llm_prompt_texts"
    __natural_key__ = ("prompt_template_id", "dataset_sample_id", "prompt_text")
    __uuid_function__ = "compute_llm_prompt_text_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    prompt_template_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.prompt_templates.id"),
        nullable=False,
    )
    dataset_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.dataset_sample.id"),
        nullable=False,
    )
    prompt_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "prompt_template_id",
            "dataset_sample_id",
            "prompt_text",
            name="uq_prompt_template_sample_text",
        ),
        {"schema": "infer"},
    )

    # Relationships
    prompt_template = relationship("PromptTemplateORM", back_populates="prompt_texts")
    llm_judgements = relationship("LLMJudgementORM", back_populates="llm_prompt_text")


class LLMResponseTextORM(Base):
    __tablename__ = "llm_response_texts"
    __natural_key__ = ("llm_response_text",)
    __uuid_function__ = "compute_llm_response_text_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    llm_response_text = Column(Text, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = {"schema": "infer"}

    # Relationships
    scores = relationship("LLMScoreORM", back_populates="llm_response_text")


class LLMInvocationMetricsORM(Base):
    __tablename__ = "llm_invocation_metrics"
    __natural_key__ = (
        "latency_ms", "retries", "cost_estimate_usd", "generation_id",
        "prompt_tokens", "completion_tokens", "total_tokens"
    )
    __uuid_function__ = "compute_llm_invocation_metrics_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    latency_ms = Column(Float, nullable=False)
    retries = Column(Integer, nullable=False, default=0)
    cost_estimate_usd = Column(Float, nullable=True)
    generation_id = Column(String(255), nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "latency_ms", "retries", "cost_estimate_usd", "generation_id",
            "prompt_tokens", "completion_tokens", "total_tokens",
            name="uq_invocation_metrics",
        ),
        {"schema": "infer"},
    )

    # Relationships
    llm_judgements = relationship("LLMJudgementORM", back_populates="llm_invocation_metrics")


class LLMScoreORM(Base):
    __tablename__ = "llm_scores"
    __natural_key__ = ("parser_spec_id", "llm_response_id")
    __uuid_function__ = "compute_llm_score_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    parser_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.parser.id"),
        nullable=False,
    )
    llm_response_text_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_response_texts.id"),
        nullable=False,
    )

    # Derived / parsed info
    label = Column(SQLEnum(RelevanceScore, schema="public"), nullable=False)
    confidence = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)

    # Parser warnings as JSONB array (data quality issues during parsing)
    parser_warnings = Column(ARRAY(JSONB), nullable=False, default=[])

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "parser_spec_id",
            "llm_response_text_id",
            name="uq_score_parser_response",
        ),
        {"schema": "infer"},
    )

    # Relationships
    parser_spec = relationship("ParserSpecORM", back_populates="scores")
    llm_response_text = relationship("LLMResponseTextORM", back_populates="scores")
    llm_judgements = relationship("LLMJudgementORM", back_populates="llm_score")


class LLMJudgementORM(Base):
    __tablename__ = "llm_judgements"
    __natural_key__ = ("judged_dataset_id", "llm_prompt_text_id")
    __uuid_function__ = "compute_llm_judgement_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    judged_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.judged_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    llm_prompt_text_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_prompt_texts.id"),
        nullable=False,
    )
    llm_invocation_metrics_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_invocation_metrics.id"),
        nullable=False,
    )
    llm_score_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_scores.id"),
        nullable=False,
    )

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "judged_dataset_id",
            "llm_prompt_text_id",
            name="uq_judgement_identity",
        ),
        {"schema": "infer"},
    )

    # Relationships
    judged_dataset = relationship("JudgedDatasetORM", back_populates="llm_judgements")
    llm_prompt_text = relationship("LLMPromptTextORM", back_populates="llm_judgements")
    llm_invocation_metrics = relationship("LLMInvocationMetricsORM", back_populates="llm_judgements")
    llm_score = relationship("LLMScoreORM", back_populates="llm_judgements")
