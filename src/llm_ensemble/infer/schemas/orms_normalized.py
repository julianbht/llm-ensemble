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


class JudgedDatasetORM(Base):
    """JudgedDataset ORM - set of LLM judgements produced by infer run.

    Represents the output of inference (LLMJudgements with scores).

    Lifecycle:
    - Created in open() with NULL fingerprint (pending state)
    - Fingerprint computed and set in close() after all judgements written
    - Enables fault-tolerant partial runs (NULL fingerprint = incomplete)

    Multiple infer runs can produce the same JudgedDataset (same fingerprint), enabling idempotency.
    Provenance to input NormalizedDataset tracked via InferRun → IngestRun → NormalizedDataset.

    Uses deterministic UUID based on fingerprint (SHA256 of sorted LLMJudgement IDs).
    """
    __tablename__ = "judged_datasets"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = ("fingerprint",)
    __uuid_function__ = "compute_judged_dataset_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    fingerprint = Column(
        CHAR(64),
        nullable=True,
        unique=True,
        comment="SHA256 of sorted LLMJudgement IDs (NULL during active run, set on completion)"
    )
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judgements = relationship(
        "LLMJudgementORM",
        secondary="infer.judged_dataset_llm_judgements",
        order_by="JudgedDatasetLLMJudgementORM.sequence_number"
    )
    infer_runs = relationship("InferRunORM", back_populates="judged_dataset")


class JudgedDatasetLLMJudgementORM(Base):
    """Junction table linking JudgedDataset to LLMJudgement with sequence.

    Preserves deterministic ordering of judgements via sequence_number.
    This enables reproducible aggregation across multiple infer runs.
    """
    __tablename__ = "judged_dataset_llm_judgements"
    __table_args__ = {"schema": "infer"}

    judged_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.judged_datasets.id", ondelete="CASCADE"),
        primary_key=True,
    )
    llm_judgement_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_judgements.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class InferRunORM(Base):
    """InferRun ORM - metadata for infer runs.

    Tracks what range of the NormalizedDataset was intended to be processed
    and what judgements were actually produced:

    Intent (set in open()):
    - start_idx, end_idx: Index range into NormalizedDataset.samples to process
    - ingest_run_id: Source dataset

    Actual result (set in close()):
    - judged_dataset_id: What judgements were actually produced (LLMCalls)
    - NULL if run incomplete/failed, populated when run completes

    Completion validation:
    - Compare judged_dataset.sample_count vs (end_idx - start_idx)
    - Enables fault tolerance and resumability

    This design enables:
    - Explicit intent: no defensive 'or' operators needed
    - Simple completion check: actual count == expected count
    - Future CLI features: --start-idx 100 --end-idx 500
    - Validation: aggregate CLI checks runs processed same samples
    """
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

    # Reference to source ingest run (provenance)
    ingest_run_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.ingest_runs.id"),
        nullable=False,
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

    # Relationships (many runs, one spec/template/model)
    model_spec = relationship("ModelSpecORM", back_populates="infer_runs")
    prompt_template = relationship("PromptTemplateORM", back_populates="infer_runs")
    judged_dataset = relationship("JudgedDatasetORM", back_populates="infer_runs")
    parser_spec = relationship("ParserSpecORM", back_populates="infer_runs")
    # Note: No explicit relationship to IngestRunORM to avoid cross-schema circular imports
    # Note: No direct relationship to LLMJudgementORM - access via judged_dataset.judgements


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

    # One parser spec can be used by many parsed scores
    scores = relationship("LLMScoreORM", back_populates="parser_spec")


class LLMPromptORM(Base):
    """LLM prompt text linked to judging sample.

    Stores the rendered prompt sent to the LLM for a specific sample.
    Deduplicated by (judging_sample_id, prompt) - same sample+prompt = same entity.
    """
    __tablename__ = "llm_prompts"
    __natural_key__ = ("prompt", "judging_sample_id")
    __uuid_function__ = "compute_llm_prompt_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id"),
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

    judgements = relationship("LLMJudgementORM", back_populates="llm_prompt")


class LLMResponseTextORM(Base):
    """Raw LLM response text.

    Stores unparsed text returned by LLM providers.
    Deduplicated by raw_response - same text = same entity.
    This enables deduplication when models return identical responses.
    """
    __tablename__ = "llm_response_texts"
    __natural_key__ = ("raw_response",)
    __uuid_function__ = "compute_llm_response_text_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    raw_response = Column(Text, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = {"schema": "infer"}

    scores = relationship("LLMScoreORM", back_populates="response_text")


class LLMInvocationMetricsORM(Base):
    """LLM invocation observability metrics.

    Captures performance and cost data from LLM API calls.
    Deduplicated by all metric fields - identical metrics = same entity.
    Enables metric reuse when calls happen to have identical performance characteristics.
    """
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

    judgements = relationship("LLMJudgementORM", back_populates="invocation_metrics")


class LLMScoreORM(Base):
    """Parsed LLM score from response text.

    Represents the result of parsing raw_response with a specific parser.
    Deduplicated by (parser_spec_id, llm_response_text_id) - same parser+text = same score.
    """
    __tablename__ = "llm_scores"
    __natural_key__ = ("parser_spec_id", "llm_response_text_id")
    __uuid_function__ = "compute_llm_score_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    parser_spec_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.parser_specs.id"),
        nullable=False,
    )
    llm_response_text_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_response_texts.id"),
        nullable=False,
    )

    # Derived / parsed info; functionally dependent on (parser_spec_id, llm_response_text_id)
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
    judgements = relationship("LLMJudgementORM", back_populates="score")
    parser_spec = relationship("ParserSpecORM", back_populates="scores")
    response_text = relationship("LLMResponseTextORM", back_populates="scores")


class LLMJudgementORM(Base):
    """Complete LLM judgement linking prompt, score, and invocation metrics.

    Represents a unique inference output by content (prompt → response → parsed score)
    plus the observed metrics from that specific API call.

    Deduplicated by content + metrics: same prompt + score + metrics = same judgement.
    This mirrors ingest pattern where JudgingSample = (query, document, gold_score).

    No direct FK to InferRun - relationship tracked via JudgedDataset.
    Multiple runs can produce the same judgement (same content/metrics).
    """
    __tablename__ = "llm_judgements"
    __natural_key__ = ("llm_prompt_id", "score_id", "llm_invocation_metrics_id")
    __uuid_function__ = "compute_llm_judgement_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)

    llm_prompt_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_prompts.id"),
        nullable=False,
    )
    score_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_scores.id"),
        nullable=False,
    )
    llm_invocation_metrics_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_invocation_metrics.id"),
        nullable=False,
    )

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "llm_prompt_id",
            "score_id",
            "llm_invocation_metrics_id",
            name="uq_judgement_content_metrics",
        ),
        {"schema": "infer"},
    )

    # Relationships
    llm_prompt = relationship("LLMPromptORM", back_populates="judgements")
    score = relationship("LLMScoreORM", back_populates="judgements")
    invocation_metrics = relationship("LLMInvocationMetricsORM", back_populates="judgements")

