"""
SQLAlchemy ORM models for INFER CLI.
Pure SQLAlchemy models for database persistence.
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

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    adapter_configs = relationship("AdapterConfigORM", back_populates="provider")


class ParserORM(Base):
    """Response text parser specification.

    Tracks which parser was used to extract structured data from LLM responses.
    Name comes from registry (e.g., 'thomas-simple').
    """
    __tablename__ = "parsers"
    __natural_key__ = "name"
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
    adapter_configs = relationship("AdapterConfigORM", back_populates="parser")


class PromptBuilderORM(Base):
    """Prompt builder with inlined template text.

    Contains both the prompt template and the builder configuration.
    """
    __tablename__ = "prompt_builders"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = "name"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    template_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    adapter_configs = relationship("AdapterConfigORM", back_populates="prompt_builder")


class AdapterConfigORM(Base):
    """Complete adapter configuration for an inference run.

    Bundles together the prompt builder, response parser, and provider.
    Multiple judged datasets can share the same adapter configuration.
    """
    __tablename__ = "adapter_configs"
    __natural_key__ = ("prompt_builder_id", "parser_id", "provider_id")
    __table_args__ = (
        UniqueConstraint(
            "prompt_builder_id",
            "parser_id",
            "provider_id",
            name="uq_adapter_config",
        ),
        {"schema": "infer"},
    )

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    prompt_builder_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.prompt_builders.id"),
        nullable=False,
    )
    parser_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.parsers.id"),
        nullable=False,
    )
    provider_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.providers.id"),
        nullable=False,
    )

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    prompt_builder = relationship("PromptBuilderORM", back_populates="adapter_configs")
    parser = relationship("ParserORM", back_populates="adapter_configs")
    provider = relationship("ProviderORM", back_populates="adapter_configs")
    judged_datasets = relationship("JudgedDatasetORM", back_populates="adapter_config")


class ModelConfigORM(Base):
    """Complete model configuration used for an inference run.

    Pure configuration data - model identity, capabilities, and inference parameters.
    """
    __tablename__ = "model_configs"
    __natural_key__ = "name"
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


class JudgedDatasetORM(Base):
    """Dataset of LLM judgements produced by an inference run.

    Combines model configuration and adapter configuration (prompt, parser, provider).
    """
    __tablename__ = "judged_datasets"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = None  # 1:1 with InferRun, uses same ID

    id = Column(PG_UUID(as_uuid=True), primary_key=True, comment="Same as InferRun.id (1:1 relationship)")

    model_config_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.model_configs.id"),
        nullable=False,
        comment="Which model configuration was used for all judgements in this dataset"
    )

    adapter_config_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.adapter_configs.id"),
        nullable=False,
        comment="Which adapter configuration (prompt builder, parser, provider) was used"
    )

    sample_fingerprint = Column(
        CHAR(64),
        nullable=True,
        comment="SHA256 of sorted dataset_sample IDs (identifies which samples were judged, for aggregation)"
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    model_config = relationship("ModelConfigORM", back_populates="judged_datasets")
    adapter_config = relationship("AdapterConfigORM", back_populates="judged_datasets")
    llm_judgements = relationship("LLMJudgementORM", back_populates="judged_dataset")
    infer_run = relationship("InferRunORM", back_populates="judged_dataset", uselist=False)


class InferRunORM(Base):
    __tablename__ = "infer_runs"
    __natural_key__ = "run_name"
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)

    # Config names snapshot for easy viewing
    config_names = Column(
        JSONB,
        nullable=False,
        comment="Config names used: {model_config, prompt_builder, parser, provider}"
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
    """Deduplicated LLM prompt texts.

    Stores unique prompt text values with no foreign keys.
    Multiple judgements can reference the same prompt text.
    """
    __tablename__ = "llm_prompt_texts"
    __natural_key__ = ("prompt_text",)
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    prompt_text = Column(Text, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    llm_judgements = relationship("LLMJudgementORM", back_populates="llm_prompt_text")


class LLMResponseTextORM(Base):
    """Deduplicated LLM response texts.

    Stores unique raw response text values.
    Multiple scores and judgements can reference the same response text.
    """
    __tablename__ = "llm_response_texts"
    __natural_key__ = ("llm_response_text",)
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    llm_response_text = Column(Text, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    llm_judgements = relationship("LLMJudgementORM", back_populates="llm_response_text")


class LLMScoreORM(Base):
    """Parsed and structured LLM score - pure value object.

    Deduplicated by content (label, confidence, rationale).
    No foreign keys - judgement connects response_text → score separately.
    """
    __tablename__ = "llm_scores"
    __natural_key__ = ("label", "confidence", "rationale")
    __table_args__ = (
        UniqueConstraint(
            "label",
            "confidence",
            "rationale",
            name="uq_score_content",
        ),
        {"schema": "infer"},
    )

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    # Parsed/derived fields
    label = Column(SQLEnum(RelevanceScore, schema="public"), nullable=False)
    confidence = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    llm_judgements = relationship("LLMJudgementORM", back_populates="llm_score")


class LLMJudgementORM(Base):
    """Single LLM judgement on a dataset sample.

    Central fact table connecting:
    - judged_dataset (which run produced this)
    - dataset_sample (what was judged)
    - llm_prompt_text (what was asked)
    - llm_response_text (what was returned)
    - llm_score (parsed result)

    Includes inlined invocation metrics (latency, tokens, cost, etc.).
    """
    __tablename__ = "llm_judgements"
    __natural_key__ = ("judged_dataset_id", "dataset_sample_id")
    __table_args__ = (
        UniqueConstraint(
            "judged_dataset_id",
            "dataset_sample_id",
            name="uq_judgement_dataset_sample",
        ),
        {"schema": "infer"},
    )

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    judged_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.judged_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.dataset_sample.id"),
        nullable=False,
        comment="Which sample from the ingest dataset was judged"
    )
    llm_prompt_text_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_prompt_texts.id"),
        nullable=False,
    )
    llm_response_text_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_response_texts.id"),
        nullable=False,
    )
    llm_score_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_scores.id"),
        nullable=False,
    )

    # Inlined invocation metrics (previously separate table)
    latency_ms = Column(Float, nullable=False)
    retries = Column(Integer, nullable=False, default=0)
    cost_estimate_usd = Column(Float, nullable=True)
    generation_id = Column(String(255), nullable=True, comment="Provider-specific generation/request ID")
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    # Parser warnings as JSONB array (data quality issues during parsing)
    parser_warnings = Column(ARRAY(JSONB), nullable=False, default=[])

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judged_dataset = relationship("JudgedDatasetORM", back_populates="llm_judgements")
    llm_prompt_text = relationship("LLMPromptTextORM", back_populates="llm_judgements")
    llm_response_text = relationship("LLMResponseTextORM", back_populates="llm_judgements")
    llm_score = relationship("LLMScoreORM", back_populates="llm_judgements")
