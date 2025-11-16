"""SQL database adapter for writing LLM judgements.

Writes LLMJudgement records to a SQL database using SQLAlchemy ORM.
This adapter decomposes the denormalized LLMJudgement domain objects
into normalized relational entities as defined in orms_normalized.py.

The adapter implements a data mapper pattern internally:
- Domain service works with LLMJudgement objects (document-oriented)
- SQL writer decomposes them into ORM entities (relational)
- Preserves hexagonal architecture by keeping this logic in the adapter

Key features:
- Run metadata (Provider, ModelSpec, PromptTemplate, ParserSpec, InferRun)
  is created once during open()
- Per-judgement data (LLMRequest, LLMResponse, LLMCall) is upserted in write_one()
- Deduplication via unique constraints + deterministic UUIDs
- Immediate commit per judgement for fault tolerance
- Handles its own logging and builds WriteSummary incrementally
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
from uuid import UUID
import structlog

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.libs.schemas.write_result import WriteResult
from llm_ensemble.libs.db import (
    get_engine,
    get_session,
    compute_provider_uuid,
    compute_model_spec_uuid,
    compute_prompt_template_uuid,
    compute_parser_spec_uuid,
    compute_infer_run_uuid,
    compute_llm_request_uuid,
    compute_llm_response_uuid,
    compute_llm_call_uuid,
)
from llm_ensemble.infer.schemas.orms_normalized import (
    ProviderORM,
    ModelSpecORM,
    PromptTemplateORM,
    ParserSpecORM,
    InferRunORM,
    LLMRequestORM,
    LLMResponseORM,
    LLMCallORM,
)
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.adapters.io.mappers import (
    provider_name_to_orm,
    model_config_to_orm,
    prompt_config_to_template_orm,
    prompt_config_to_parser_orm,
    infer_run_info_to_orm,
)
from llm_ensemble.libs.logging.log_events import InferWriteEvent


class SqlJudgementWriter(JudgementWriter):
    """Write LLMJudgement records to SQL database with normalized schema.

    This adapter implements the JudgementWriter port while handling the
    impedance mismatch between domain objects and relational entities.

    Architecture:
    - Data mapper logic lives inside this adapter (preserves domain purity)
    - Run metadata initialized once in open()
    - Per-judgement decomposition in write_one()

    Database connection:
    - Reads DATABASE_URL from environment (.env file)
    - Uses SQLAlchemy session with autocommit=False
    - Commits after each write_one() for fault tolerance

    Deduplication strategy:
    - Deterministic UUIDs ensure same logical entity → same UUID
    - Unique constraints on natural keys prevent duplicates
    - merge() handles insert-or-ignore logic
    """

    def __init__(self):
        """Initialize SQL writer with its own logger."""
        super().__init__()
        self._session: Optional[Session] = None
        self._run_dir: Optional[Path] = None

        # Cached IDs from run metadata initialization
        self._provider_id: Optional[UUID] = None
        self._model_spec_id: Optional[UUID] = None
        self._prompt_template_id: Optional[UUID] = None
        self._parser_spec_id: Optional[UUID] = None
        self._infer_run_id: Optional[UUID] = None

        # Incremental write summary builder
        self._write_summary: Optional[WriteSummary] = None

        # Logger for this adapter
        self.logger = structlog.get_logger().bind(component="sql_judgement_writer")

    def open(self, run_dir: Path, run_info: InferRunInfo) -> "SqlJudgementWriter":
        """Open database session and initialize run metadata.

        This method:
        1. Creates SQLAlchemy session from DATABASE_URL env var
        2. Immediately initializes run metadata (Provider, ModelSpec, PromptTemplate, etc.)
        3. Ready for streaming write_one() calls

        Args:
            run_dir: Run directory (cached but not used for DB writer)
            run_info: Inference run context (used to create run metadata entities)

        Returns:
            Self, to enable context manager usage

        Raises:
            ValueError: If DATABASE_URL environment variable is not set
            RuntimeError: If writer is already open
        """
        if self._session is not None:
            raise RuntimeError("Writer is already open")

        # Get database engine from environment
        engine = get_engine()  # Reads DATABASE_URL from .env
        self._session = get_session(engine)

        # Cache run_dir (not used for DB, but kept for consistency)
        self._run_dir = run_dir

        # Initialize WriteSummary builder
        self._write_summary = WriteSummary()

        # Initialize run metadata immediately using run_info (logs and adds to summary)
        self._initialize_run_metadata(run_info)

        return self

    def write_one(self, judgement: LLMJudgement) -> WriteResult:
        """Write a single judgement to database.

        Decomposes LLMJudgement into normalized ORM entities:
        1. Upsert LLMRequestORM (deduplicated by prompt + sample)
        2. Upsert LLMResponseORM (deduplicated by parser + raw_response)
        3. Create LLMCallORM (links request + response + run)
        4. Commit transaction immediately

        Run metadata was already initialized in open().

        Args:
            judgement: LLMJudgement object to write

        Returns:
            WriteResult with the LLMCall UUID

        Raises:
            RuntimeError: If called outside of context manager
            IntegrityError: If database constraints are violated
        """
        if self._session is None:
            raise RuntimeError("Writer is not open - must call within context manager")

        # Decompose judgement into ORM entities with tracking
        request_id, req_created, req_skipped = self._upsert_request(judgement)
        self._write_summary.add_llm_requests(created=req_created, skipped=req_skipped)

        response_id, resp_created, resp_skipped = self._upsert_response(judgement)
        self._write_summary.add_llm_responses(created=resp_created, skipped=resp_skipped)

        call_id = self._create_call(judgement, request_id, response_id)
        self._write_summary.add_llm_calls(created=1)

        # Commit transaction (fault tolerance - each judgement is persisted immediately)
        self._session.commit()

        # Return result for this specific write
        return WriteResult(
            item_id=call_id,
            item_type="llm_call"
        )

    def close(self) -> WriteSummary:
        """Close database session and release resources.

        Returns:
            WriteSummary with aggregate statistics

        Raises:
            IOError: If close operation fails
        """
        # Get summary before cleanup
        summary = self._write_summary

        # Log final totals
        if summary and (summary.total_created > 0 or summary.total_skipped > 0):
            self.logger.info(
                InferWriteEvent.WRITE_COMPLETE,
                total_created=summary.total_created,
                total_skipped=summary.total_skipped,
            )

        if self._session is not None:
            self._session.close()
            self._session = None
            self._run_dir = None

            # Clear cached IDs
            self._provider_id = None
            self._model_spec_id = None
            self._prompt_template_id = None
            self._parser_spec_id = None
            self._infer_run_id = None
            self._write_summary = None

        return summary

    # ========================================================================
    # Internal Data Mapper Methods
    # ========================================================================

    def _initialize_run_metadata(self, run_info: InferRunInfo) -> None:
        """Initialize run metadata from run context with logging and tracking.

        Creates/upserts shared entities that remain constant across all judgements:
        - Provider (e.g., openrouter, ollama)
        - ModelSpec (model config with inference parameters)
        - PromptTemplate (prompt config with template text)
        - ParserSpec (parser adapter specification)
        - InferRun (run metadata and parameters)

        Args:
            run_info: Inference run context (passed to open())
        """

        # 1. Upsert Provider
        self._provider_id, created, skipped = self._upsert_provider(run_info.model_cfg.provider)
        self._write_summary.add_providers(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info("write_providers", created=created, skipped=skipped)

        # 2. Upsert ModelSpec (depends on Provider)
        self._model_spec_id, created, skipped = self._upsert_model_spec(run_info.model_cfg)
        self._write_summary.add_model_specs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info("write_model_specs", created=created, skipped=skipped)

        # 3. Upsert PromptTemplate
        self._prompt_template_id, created, skipped = self._upsert_prompt_template(run_info.prompt_config)
        self._write_summary.add_prompt_templates(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info("write_prompt_templates", created=created, skipped=skipped)

        # 4. Upsert ParserSpec
        self._parser_spec_id, created, skipped = self._upsert_parser_spec(run_info.prompt_config)
        self._write_summary.add_parser_specs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info("write_parser_specs", created=created, skipped=skipped)

        # 5. Create InferRun (depends on all above)
        self._infer_run_id, created, skipped = self._create_infer_run(run_info)
        self._write_summary.add_infer_runs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info("write_infer_runs", created=created, skipped=skipped)

        # Commit run metadata
        self._session.commit()

    def _upsert_provider(self, provider_name: str) -> Tuple[UUID, int, int]:
        """Upsert provider entity using mapper.

        Args:
            provider_name: Provider name (e.g., 'openrouter', 'ollama', 'hf')

        Returns:
            Tuple of (provider_id, created_count, skipped_count)
        """
        provider_id = compute_provider_uuid(provider_name)

        # Check if exists
        existing = self._session.get(ProviderORM, provider_id)
        if existing:
            return (provider_id, 0, 1)

        # Create new provider using mapper
        provider_orm = provider_name_to_orm(provider_name)
        self._session.add(provider_orm)

        return (provider_id, 1, 0)

    def _upsert_model_spec(self, model_cfg: ModelConfig) -> Tuple[UUID, int, int]:
        """Upsert model spec entity using mapper.

        Args:
            model_cfg: ModelConfig object from InferRunInfo

        Returns:
            Tuple of (model_spec_id, created_count, skipped_count)
        """
        model_spec_id = compute_model_spec_uuid(model_cfg.name)

        # Check if exists
        existing = self._session.get(ModelSpecORM, model_spec_id)
        if existing:
            return (model_spec_id, 0, 1)

        # Create new model spec using mapper
        model_spec_orm = model_config_to_orm(model_cfg, self._provider_id)
        self._session.add(model_spec_orm)

        return (model_spec_id, 1, 0)

    def _upsert_prompt_template(self, prompt_cfg: PromptConfig) -> Tuple[UUID, int, int]:
        """Upsert prompt template entity using mapper.

        Args:
            prompt_cfg: PromptConfig object from InferRunInfo

        Returns:
            Tuple of (prompt_template_id, created_count, skipped_count)
        """
        prompt_template_id = compute_prompt_template_uuid(prompt_cfg.name)

        # Check if exists
        existing = self._session.get(PromptTemplateORM, prompt_template_id)
        if existing:
            return (prompt_template_id, 0, 1)

        # Load template text from the builder
        # The builder knows where to find its template
        builder = prompt_cfg.get_prompt_builder()
        template_text = getattr(builder, "template_text", "")  # Get template text if available

        # Create new prompt template using mapper
        prompt_template_orm = prompt_config_to_template_orm(prompt_cfg, template_text)
        self._session.add(prompt_template_orm)

        return (prompt_template_id, 1, 0)

    def _upsert_parser_spec(self, prompt_cfg: PromptConfig) -> Tuple[UUID, int, int]:
        """Upsert parser spec entity using mapper.

        Args:
            prompt_cfg: PromptConfig object from InferRunInfo

        Returns:
            Tuple of (parser_spec_id, created_count, skipped_count)
        """
        # Use dummy code_hash for now (as agreed with user)
        code_hash = "0" * 64  # Placeholder - will be computed properly later

        parser_spec_id = compute_parser_spec_uuid(
            prompt_cfg.parser_module,
            prompt_cfg.parser_class,
            code_hash
        )

        # Check if exists
        existing = self._session.get(ParserSpecORM, parser_spec_id)
        if existing:
            return (parser_spec_id, 0, 1)

        # Create new parser spec using mapper
        parser_spec_orm = prompt_config_to_parser_orm(prompt_cfg, code_hash)
        self._session.add(parser_spec_orm)

        return (parser_spec_id, 1, 0)

    def _create_infer_run(self, run_info: InferRunInfo) -> Tuple[UUID, int, int]:
        """Create infer run entity using mapper.

        Args:
            run_info: InferRunInfo object from judgement

        Returns:
            Tuple of (infer_run_id, created_count, skipped_count)
        """
        infer_run_id = compute_infer_run_uuid(run_info.run_name)

        # Check if exists (should not happen, but defensive)
        existing = self._session.get(InferRunORM, infer_run_id)
        if existing:
            return (infer_run_id, 0, 1)

        # Create new infer run using mapper
        infer_run_orm = infer_run_info_to_orm(
            run_info,
            self._model_spec_id,
            self._prompt_template_id,
            self._parser_spec_id,
        )
        self._session.add(infer_run_orm)

        return (infer_run_id, 1, 0)

    def _upsert_request(self, judgement: LLMJudgement) -> Tuple[UUID, int, int]:
        """Upsert LLM request entity.

        Args:
            judgement: LLMJudgement object

        Returns:
            Tuple of (request_id, created_count, skipped_count)
        """
        request_id = compute_llm_request_uuid(
            judgement.prompt,
            judgement.judging_sample.id
        )

        # Check if exists (deduplication)
        existing = self._session.get(LLMRequestORM, request_id)
        if existing:
            return (request_id, 0, 1)

        # Create new request
        request = LLMRequestORM(
            id=request_id,
            judging_sample_id=judgement.judging_sample.id,
            prompt=judgement.prompt,
        )
        self._session.add(request)

        return (request_id, 1, 0)

    def _upsert_response(self, judgement: LLMJudgement) -> Tuple[UUID, int, int]:
        """Upsert LLM response entity.

        Deduplicates by (parser_spec_id, raw_response).
        Parsed fields (label, confidence, rationale) are functionally dependent
        on this key, so they're stored with the response.

        Args:
            judgement: LLMJudgement object

        Returns:
            Tuple of (response_id, created_count, skipped_count)
        """
        response_id = compute_llm_response_uuid(
            self._parser_spec_id,
            judgement.llm_response.raw_response
        )

        # Check if exists (deduplication)
        existing = self._session.get(LLMResponseORM, response_id)
        if existing:
            return (response_id, 0, 1)

        # Extract parsed fields from llm_score (may be None if parsing failed)
        label = judgement.llm_score.label if judgement.llm_score else None
        confidence = judgement.llm_score.confidence if judgement.llm_score else None
        rationale = judgement.llm_score.rationale if judgement.llm_score else None

        # Extract parser warnings (convert to dict format for JSONB array)
        parser_warnings = []
        if judgement.llm_score:
            parser_warnings = [w.to_dict() for w in judgement.llm_score.warnings]

        # Handle case where label is None (parsing failed) - need default for non-nullable column
        # The ORM defines label as non-nullable, so we need a default
        # Use RelevanceScore.NOT_RELEVANT as a safe default for failed parsing
        if label is None:
            from llm_ensemble.libs.schemas.relevance_score import RelevanceScore
            label = RelevanceScore.NOT_RELEVANT  # Default for parsing failures

        # Create new response
        response = LLMResponseORM(
            id=response_id,
            parser_spec_id=self._parser_spec_id,
            raw_response=judgement.llm_response.raw_response,
            label=label,
            confidence=confidence,
            rationale=rationale,
            parser_warnings=parser_warnings,
        )
        self._session.add(response)

        return (response_id, 1, 0)

    def _create_call(self, judgement: LLMJudgement, request_id: UUID, response_id: UUID) -> UUID:
        """Create LLM call entity.

        Links request + run + response with observability metadata.

        Args:
            judgement: LLMJudgement object
            request_id: UUID of the request
            response_id: UUID of the response

        Returns:
            LLMCall UUID (deterministic)
        """
        call_id = compute_llm_call_uuid(request_id, self._infer_run_id)

        # Check if exists (should not happen due to unique constraint, but defensive)
        existing = self._session.get(LLMCallORM, call_id)
        if existing:
            return call_id

        # Create new call
        call = LLMCallORM(
            id=call_id,
            llm_request_id=request_id,
            infer_run_id=self._infer_run_id,
            response_id=response_id,
            latency_ms=judgement.llm_response.latency_ms,
            retries=judgement.llm_response.retries,
            cost_estimate_usd=judgement.llm_response.cost_estimate_usd,
        )
        self._session.add(call)

        return call_id
