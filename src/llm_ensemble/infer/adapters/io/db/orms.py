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
    infer_run_configs = relationship("InferRunConfigORM", back_populates="provider")


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
    prompt_templates = relationship("PromptTemplateORM", back_populates="parser")


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
    prompt_templates = relationship("PromptTemplateORM", back_populates="prompt_builder")


class PromptTemplateORM(Base):
    """Prompt template that bundles prompt builder and parser.

    Represents a complete prompt template with both builder and parser metadata.
    The template_text is stored on PromptBuilderORM to avoid duplication.
    """
    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint(
            "name",
            name="uq_prompt_template_name",
        ),
        {"schema": "infer"},
    )
    __natural_key__ = "name"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True, comment="Template name (e.g., 'thomas-simple')")

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

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    prompt_builder = relationship("PromptBuilderORM", back_populates="prompt_templates")
    parser = relationship("ParserORM", back_populates="prompt_templates")
    infer_run_configs = relationship("InferRunConfigORM", back_populates="prompt_template")


class IngestRunContextORM(Base):
    """Execution context for an infer run.

    Captures which ingest run to read from and which samples to process.
    Separate table allows deduplication when multiple runs use the same context.
    """
    __tablename__ = "ingest_run_contexts"
    __natural_key__ = ("input_run_name", "start_idx", "end_idx")
    __table_args__ = (
        UniqueConstraint(
            "input_run_name",
            "start_idx",
            "end_idx",
            name="uq_ingest_run_context",
        ),
        {"schema": "infer"},
    )

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    input_run_name = Column(String(255), nullable=False, comment="Ingest run name to read samples from")
    start_idx = Column(Integer, nullable=True, comment="Start index into NormalizedDataset.samples (None = from beginning)")
    end_idx = Column(Integer, nullable=True, comment="End index into NormalizedDataset.samples (None = until end)")
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    infer_run_configs = relationship("InferRunConfigORM", back_populates="ingest_run_context")


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
    infer_run_configs = relationship("InferRunConfigORM", back_populates="model_config")


class InferRunConfigORM(Base):
    """Complete configuration bundle for an inference run.

    Bundles all configuration needed to execute inference:
    - Model configuration
    - Provider configuration
    - Prompt template (builder + parser)
    - Execution context (input source, sample range)

    Note: retry_config is not persisted (transient execution detail).
    """
    __tablename__ = "infer_run_configs"
    __table_args__ = (
        UniqueConstraint(
            "model_config_id",
            "provider_id",
            "prompt_template_id",
            "ingest_run_context_id",
            name="uq_infer_run_config",
        ),
        {"schema": "infer"},
    )
    __natural_key__ = ("model_config_id", "provider_id", "prompt_template_id", "ingest_run_context_id")

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    model_config_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.model_configs.id"),
        nullable=False,
    )
    provider_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.providers.id"),
        nullable=False,
    )
    prompt_template_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.prompt_templates.id"),
        nullable=False,
    )
    ingest_run_context_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.ingest_run_contexts.id"),
        nullable=False,
    )

    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    model_config = relationship("ModelConfigORM", back_populates="infer_run_configs")
    provider = relationship("ProviderORM", back_populates="infer_run_configs")
    prompt_template = relationship("PromptTemplateORM", back_populates="infer_run_configs")
    ingest_run_context = relationship("IngestRunContextORM", back_populates="infer_run_configs")
    infer_run_outputs = relationship("InferRunOutputORM", back_populates="infer_run_config")


class InferRunOutputORM(Base):
    """Output produced during an inference run.

    Contains:
    - Reference to the complete configuration used (InferRunConfigORM)
    - Sample fingerprint identifying which samples were judged
    - LLM judgements (via relationship)

    This is the "what was produced" entity, linking to "what configuration was used".
    """
    __tablename__ = "infer_run_outputs"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = None  # 1:1 with InferRun, uses same ID

    id = Column(PG_UUID(as_uuid=True), primary_key=True, comment="Same as InferRun.id (1:1 relationship)")

    infer_run_config_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.infer_run_configs.id"),
        nullable=False,
        comment="Complete configuration bundle used to produce these judgements"
    )

    sample_fingerprint = Column(
        CHAR(64),
        nullable=True,
        comment="SHA256 of sorted dataset_sample IDs (identifies which samples were judged, for aggregation)"
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    infer_run_config = relationship("InferRunConfigORM", back_populates="infer_run_outputs")
    llm_judgements = relationship("LLMJudgementORM", back_populates="infer_run_output")
    infer_run = relationship("InferRunORM", back_populates="infer_run_output", uselist=False)


class InferRunORM(Base):
    __tablename__ = "infer_runs"
    __natural_key__ = "run_name"
    __table_args__ = {"schema": "infer"}

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    run_name = Column(String(255), nullable=False, unique=True)
    run_type = Column(SQLEnum(RunType, schema="public"), nullable=False, default=RunType.TEST)

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

    # ACTUAL RESULT: What was actually produced (set in close())
    infer_run_output_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.infer_run_outputs.id"),
        nullable=True,
        comment="What output was actually produced (NULL = run incomplete/failed)"
    )

    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    infer_run_output = relationship("InferRunOutputORM", back_populates="infer_run", uselist=False)


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
    - infer_run_output (which run produced this)
    - dataset_sample (what was judged)
    - llm_prompt_text (what was asked)
    - llm_response_text (what was returned)
    - llm_score (parsed result)

    Includes inlined invocation metrics (latency, tokens, cost, etc.).
    """
    __tablename__ = "llm_judgements"
    __natural_key__ = ("infer_run_output_id", "dataset_sample_id")
    __table_args__ = (
        UniqueConstraint(
            "infer_run_output_id",
            "dataset_sample_id",
            name="uq_judgement_output_sample",
        ),
        {"schema": "infer"},
    )

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    infer_run_output_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.infer_run_outputs.id", ondelete="CASCADE"),
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
    infer_run_output = relationship("InferRunOutputORM", back_populates="llm_judgements")
    llm_prompt_text = relationship("LLMPromptTextORM", back_populates="llm_judgements")
    llm_response_text = relationship("LLMResponseTextORM", back_populates="llm_judgements")
    llm_score = relationship("LLMScoreORM", back_populates="llm_judgements")
