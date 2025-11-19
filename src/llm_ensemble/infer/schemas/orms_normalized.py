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


class InferredDatasetORM(Base):
    """InferredDataset ORM - set of samples actually processed by infer run.

    Represents the working set for inference. Multiple infer runs can produce
    the same InferredDataset (same fingerprint), enabling idempotency.

    Provenance to NormalizedDataset is tracked via InferRun → IngestRun → NormalizedDataset.

    Uses deterministic UUID based on fingerprint (SHA256 of sorted sample IDs).
    """
    __tablename__ = "inferred_datasets"
    __table_args__ = {"schema": "infer"}
    __natural_key__ = ("fingerprint",)
    __uuid_function__ = "compute_normalized_dataset_uuid"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    fingerprint = Column(CHAR(64), nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships
    judging_samples = relationship(
        "JudgingSampleORM",
        secondary="infer.inferred_dataset_judging_samples",
        order_by="InferredDatasetJudgingSampleORM.sequence_number"
    )
    infer_runs = relationship("InferRunORM", back_populates="inferred_dataset")


class InferredDatasetJudgingSampleORM(Base):
    """Junction table linking InferredDataset to JudgingSample with sequence.

    Preserves deterministic ordering of samples via sequence_number.
    This enables reproducible aggregation across multiple infer runs.
    """
    __tablename__ = "inferred_dataset_judging_samples"
    __table_args__ = {"schema": "infer"}

    inferred_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.inferred_datasets.id", ondelete="CASCADE"),
        primary_key=True,
    )
    judging_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class InferRunORM(Base):
    """InferRun ORM - metadata for infer runs.

    Separates user intent (what was requested) from actual result (what was inferred):
    
    Intent (set in open()):
    - start_sample_id, end_sample_id: Requested sample range to process
    - limit: Optional limit on number of samples
    - ingest_run_id: Source dataset
    
    Actual result (set in close()):
    - inferred_dataset_id: What samples actually got judgements
    - NULL if run incomplete/failed, populated when run completes successfully
    
    This design enables:
    - Fault tolerance: partial runs persist with incomplete state
    - Resumability: compare intent vs actual to find missing samples
    - Validation: aggregate CLI checks all runs share same inferred_dataset_id
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

    # INTENT: What user requested to process (set in open())
    start_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id"),
        nullable=True,
        comment="First sample user intended to process (NULL = start from beginning)"
    )
    end_sample_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ingest.judging_samples.id"),
        nullable=True,
        comment="Last sample user intended to process (NULL = process until end)"
    )
    limit = Column(Integer, nullable=True, comment="Maximum number of samples to process")

    # ACTUAL RESULT: What was actually inferred (set in close())
    inferred_dataset_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.inferred_datasets.id"),
        nullable=True,
        comment="What samples actually got judgements (NULL = run incomplete/failed)"
    )

    git_sha = Column(String(40), nullable=True)
    git_branch = Column(String(255), nullable=True)
    git_is_dirty = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    # Relationships (many runs, one spec/template/model)
    model_spec = relationship("ModelSpecORM", back_populates="infer_runs")
    prompt_template = relationship("PromptTemplateORM", back_populates="infer_runs")
    inferred_dataset = relationship("InferredDatasetORM", back_populates="infer_runs")
    parser_spec = relationship("ParserSpecORM", back_populates="infer_runs")
    calls = relationship("LLMCallORM", back_populates="infer_run")
    # Note: No explicit relationship to IngestRunORM to avoid cross-schema circular imports


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


class LLMRequestORM(Base):
    __tablename__ = "llm_requests"
    __natural_key__ = ("prompt", "judging_sample_id")
    __uuid_function__ = "compute_llm_request_uuid"

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

    # Each call has at most one (possibly shared) score
    score_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("infer.llm_scores.id"),
        nullable=True,
    )

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
            "llm_request_id",
            "infer_run_id",
            name="uq_call_per_run_and_request",
        ),
        {"schema": "infer"},
    )

    llm_request = relationship("LLMRequestORM", back_populates="calls")
    infer_run = relationship("InferRunORM", back_populates="calls")
    score = relationship("LLMScoreORM", back_populates="calls")


class LLMScoreORM(Base):
    __tablename__ = "llm_scores"
    __natural_key__ = ("parser_spec_id", "raw_response")
    __uuid_function__ = "compute_llm_score_uuid"

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
    parser_warnings = Column(ARRAY(JSONB), nullable=False, default=[])

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "parser_spec_id",
            "raw_response",
            name="uq_score_parser_raw",
        ),
        {"schema": "infer"},
    )

    # One score can be reused by many calls
    calls = relationship("LLMCallORM", back_populates="score")
    parser_spec = relationship("ParserSpecORM", back_populates="scores")

