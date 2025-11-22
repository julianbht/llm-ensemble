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
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.db import (
    get_engine,
    get_session,
    compute_provider_uuid,
    compute_model_spec_uuid,
    compute_prompt_template_uuid,
    compute_parser_spec_uuid,
    compute_infer_run_uuid,
    compute_llm_request_uuid,
    compute_llm_score_uuid,
    compute_llm_call_uuid,
    compute_judged_dataset_uuid,
    compute_normalized_dataset_uuid,
)
from llm_ensemble.infer.schemas.orms_normalized import (
    ProviderORM,
    ModelSpecORM,
    PromptTemplateORM,
    ParserSpecORM,
    InferRunORM,
    JudgedDatasetORM,
    JudgedDatasetLLMCallORM,
    LLMRequestORM,
    LLMScoreORM,
    LLMCallORM,
)
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.adapters.io.mappers_domain_to_orm import (
    provider_name_to_orm,
    model_config_to_orm,
    prompt_config_to_template_orm,
    prompt_config_to_parser_orm,
    infer_run_info_to_orm,
    llm_judgement_to_request_orm,
    llm_judgement_to_score_orm,
    llm_judgement_to_call_orm,
)
from llm_ensemble.libs.logging.log_events import InferWriteEvent, InferLogEvent


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
        self._judged_dataset_id: Optional[UUID] = None

        # Accumulate LLMCall IDs for JudgedDataset fingerprint computation
        self._call_ids: list[UUID] = []

        # Incremental write summary builder
        self._write_summary: Optional[WriteSummary] = None

        # Logger for this adapter (includes CLI context from orchestrator)
        self.logger = get_logger(component="sql_judgement_writer")

    def open(
        self,
        run_dir: Path,
        run_info: InferRunInfo,
        normalized_dataset: NormalizedDataset,
        start_idx: int,
        end_idx: int,
    ) -> "SqlJudgementWriter":
        """Open database session and initialize run metadata.

        This method:
        1. Creates SQLAlchemy session from DATABASE_URL env var
        2. Initializes run metadata (Provider, ModelSpec, PromptTemplate, InferRun with indices)
        3. Creates JudgedDataset entity with NULL fingerprint (pending state)
        4. Ready for streaming write_one() calls

        Args:
            run_dir: Run directory (cached but not used for DB writer)
            run_info: Inference run context with nullable indices capturing user intent
            normalized_dataset: Input dataset (used for provenance linking)
            start_idx: Computed start index (defaults to 0 if user didn't specify)
            end_idx: Computed end index (defaults to sample_count if user didn't specify)

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

        # Initialize run metadata with computed indices (logs and adds to summary)
        self._initialize_run_metadata(run_info, normalized_dataset, start_idx, end_idx)

        # Create JudgedDataset entity with NULL fingerprint (finalized in close())
        self._create_judged_dataset_entity()

        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to database.

        Decomposes LLMJudgement into normalized ORM entities:
        1. Upsert LLMRequestORM (deduplicated by prompt + sample)
        2. Upsert LLMScoreORM (deduplicated by parser + raw_response)
        3. Create LLMCallORM (links request + score + run)
        4. Commit transaction immediately
        5. Log all entities written in one line

        Run metadata was already initialized in open().

        Args:
            judgement: LLMJudgement object to write

        Raises:
            RuntimeError: If called outside of context manager
            IntegrityError: If database constraints are violated
        """
        if self._session is None:
            raise RuntimeError("Writer is not open - must call within context manager")

        # Decompose judgement into ORM entities with tracking and logging
        request_id, req_created, req_skipped = self._upsert_request(judgement)
        self._write_summary.add_llm_requests(created=req_created, skipped=req_skipped)
        if req_created > 0 or req_skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_LLM_REQUESTS, created=req_created, skipped=req_skipped)

        score_id, score_created, score_skipped = self._upsert_score(judgement)
        self._write_summary.add_llm_scores(created=score_created, skipped=score_skipped)
        if score_created > 0 or score_skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_LLM_SCORES, created=score_created, skipped=score_skipped)

        call_id = self._create_call(judgement, request_id, score_id)
        self._write_summary.add_llm_calls(created=1)
        self.logger.info(InferWriteEvent.WRITE_LLM_CALLS, created=1, skipped=0)

        # Accumulate LLMCall ID for JudgedDataset fingerprint computation
        self._call_ids.append(call_id)

        # Commit transaction (fault tolerance - each judgement is persisted immediately)
        self._session.commit()

    def close(self) -> WriteSummary:
        """Close database session and release resources.

        Finalizes the inference run by:
        1. Computing JudgedDataset fingerprint from accumulated LLMCall IDs
        2. Upserting JudgedDataset entity and junction records
        3. Updating InferRun with judged_dataset_id (marks run as complete)
        4. Committing final transaction

        Returns:
            WriteSummary with aggregate statistics

        Raises:
            IOError: If close operation fails
        """
        # Finalize JudgedDataset before cleanup
        if self._session is not None and self._call_ids:
            judged_dataset_id = self._finalize_judged_dataset()

            # Update InferRun with actual judged_dataset_id (marks completion)
            infer_run = self._session.get(InferRunORM, self._infer_run_id)
            if infer_run:
                infer_run.judged_dataset_id = judged_dataset_id
                self._session.commit()

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
            self._judged_dataset_id = None
            self._call_ids = []
            self._write_summary = None

        return summary

    # ========================================================================
    # Internal Data Mapper Methods
    # ========================================================================

    def _initialize_run_metadata(
        self,
        run_info: InferRunInfo,
        normalized_dataset: NormalizedDataset,
        start_idx: int,
        end_idx: int
    ) -> None:
        """Initialize run metadata from run context with logging and tracking.

        Creates/upserts shared entities that remain constant across all judgements:
        - Provider (e.g., openrouter, ollama)
        - ModelSpec (model config with inference parameters)
        - PromptTemplate (prompt config with template text)
        - ParserSpec (parser adapter specification)
        - InferRun (run metadata with explicit start_idx/end_idx)

        Args:
            run_info: Inference run context (passed to open())
            normalized_dataset: Input dataset (for provenance linking)
            start_idx: Start index into NormalizedDataset.samples
            end_idx: End index into NormalizedDataset.samples
        """

        # 1. Upsert Provider
        self._provider_id, created, skipped = self._upsert_provider(run_info.model_cfg.provider)
        self._write_summary.add_providers(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_PROVIDERS, created=created, skipped=skipped)

        # 2. Upsert ModelSpec (depends on Provider)
        self._model_spec_id, created, skipped = self._upsert_model_spec(run_info.model_cfg)
        self._write_summary.add_model_specs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_MODEL_SPECS, created=created, skipped=skipped)

        # 3. Upsert PromptTemplate
        self._prompt_template_id, created, skipped = self._upsert_prompt_template(run_info.prompt_config)
        self._write_summary.add_prompt_templates(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_PROMPT_TEMPLATES, created=created, skipped=skipped)

        # 4. Upsert ParserSpec
        self._parser_spec_id, created, skipped = self._upsert_parser_spec(run_info.prompt_config)
        self._write_summary.add_parser_specs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_PARSER_SPECS, created=created, skipped=skipped)

        # 5. Create InferRun with explicit indices (depends on all above)
        self._infer_run_id, created, skipped = self._create_infer_run(
            run_info, normalized_dataset, start_idx, end_idx
        )
        self._write_summary.add_infer_runs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_INFER_RUNS, created=created, skipped=skipped)

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

    def _create_infer_run(
        self,
        run_info: InferRunInfo,
        normalized_dataset: NormalizedDataset,
        start_idx: int,
        end_idx: int
    ) -> Tuple[UUID, int, int]:
        """Create infer run entity using mapper.

        Args:
            run_info: InferRunInfo object
            normalized_dataset: Input dataset (for ingest_run_id)
            start_idx: Computed start index (actual value processed)
            end_idx: Computed end index (actual value processed)

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
            normalized_dataset.id,  # ingest_run_id for provenance
            start_idx,
            end_idx,
        )
        self._session.add(infer_run_orm)

        return (infer_run_id, 1, 0)

    def _upsert_request(self, judgement: LLMJudgement) -> Tuple[UUID, int, int]:
        """Upsert LLM request entity using mapper.

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

        # Create new request using mapper
        request = llm_judgement_to_request_orm(judgement, request_id)
        self._session.add(request)

        return (request_id, 1, 0)

    def _upsert_score(self, judgement: LLMJudgement) -> Tuple[UUID, int, int]:
        """Upsert LLM score entity using mapper.

        Deduplicates by (parser_spec_id, raw_response).
        Parsed fields (label, confidence, rationale) are functionally dependent
        on this key, so they're stored with the score.

        Args:
            judgement: LLMJudgement object

        Returns:
            Tuple of (score_id, created_count, skipped_count)
        """
        score_id = compute_llm_score_uuid(
            self._parser_spec_id,
            judgement.llm_response.raw_response
        )

        # Check if exists (deduplication)
        existing = self._session.get(LLMScoreORM, score_id)
        if existing:
            return (score_id, 0, 1)

        # Create new score using mapper
        score = llm_judgement_to_score_orm(
            judgement,
            score_id,
            self._parser_spec_id
        )
        self._session.add(score)

        return (score_id, 1, 0)

    def _create_call(self, judgement: LLMJudgement, request_id: UUID, score_id: UUID) -> UUID:
        """Create LLM call entity using mapper.

        Links request + run + score with observability metadata.

        Args:
            judgement: LLMJudgement object
            request_id: UUID of the request
            score_id: UUID of the score

        Returns:
            LLMCall UUID (deterministic)
        """
        call_id = compute_llm_call_uuid(request_id, self._infer_run_id)

        # Check if exists (should not happen due to unique constraint, but defensive)
        existing = self._session.get(LLMCallORM, call_id)
        if existing:
            return call_id

        # Create new call using mapper
        call = llm_judgement_to_call_orm(
            judgement,
            call_id,
            request_id,
            self._infer_run_id,
            score_id
        )
        self._session.add(call)

        return call_id

    def _finalize_judged_dataset(self) -> UUID:
        """Finalize JudgedDataset after all judgements written.

        Computes fingerprint from accumulated LLMCall IDs and updates the
        existing JudgedDataset entity (created in open()). Also creates
        junction records linking calls to dataset.

        Returns:
            UUID of the JudgedDataset

        Raises:
            ValueError: If no calls were processed
        """
        if not self._call_ids:
            raise ValueError("Cannot finalize JudgedDataset - no calls processed")

        # Sort LLMCall IDs for deterministic fingerprint
        sorted_call_ids = sorted(self._call_ids)

        # Compute fingerprint from sorted UUIDs
        id_string = ",".join(str(cid) for cid in sorted_call_ids)
        fingerprint = hashlib.sha256(id_string.encode()).hexdigest()

        # Update existing JudgedDataset with computed fingerprint
        judged_dataset = self._session.get(JudgedDatasetORM, self._judged_dataset_id)
        if not judged_dataset:
            raise RuntimeError(f"JudgedDataset {self._judged_dataset_id} not found - should have been created in open()")

        judged_dataset.fingerprint = fingerprint
        self._session.flush()

        self.logger.info(InferWriteEvent.WRITE_JUDGED_DATASETS, created=0, skipped=0, updated=1)

        # Create junction records linking dataset to LLMCalls (with sequence numbers)
        junction_created = self._create_judged_dataset_junctions(self._judged_dataset_id, sorted_call_ids)
        self._write_summary.add_judged_dataset_junctions(created=junction_created)
        if junction_created > 0:
            self.logger.info(InferWriteEvent.WRITE_JUDGED_DATASET_JUNCTIONS, created=junction_created)

        return self._judged_dataset_id

    def _create_judged_dataset_entity(self) -> None:
        """Create JudgedDataset entity with NULL fingerprint (pending state).

        Called in open() to create the dataset entity early. The fingerprint
        is computed and set in close() after all judgements are written.

        The UUID is generated from a temporary placeholder fingerprint and
        will be recomputed when the actual fingerprint is known.
        """
        import uuid

        # Generate temporary UUID for this pending dataset
        # This will remain the ID throughout the run
        self._judged_dataset_id = uuid.uuid4()

        # Create JudgedDataset with NULL fingerprint (pending completion)
        dataset_orm = JudgedDatasetORM(
            id=self._judged_dataset_id,
            fingerprint=None,  # Will be set in close()
        )
        self._session.add(dataset_orm)
        self._session.flush()

        # Track creation in summary
        self._write_summary.add_judged_datasets(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_JUDGED_DATASETS, created=1, skipped=0)

    def _create_judged_dataset_junctions(
        self, dataset_id: UUID, sorted_call_ids: list[UUID]
    ) -> int:
        """Create junction records linking JudgedDataset to LLMCalls.

        Args:
            dataset_id: JudgedDataset UUID
            sorted_call_ids: Sorted list of LLMCall UUIDs

        Returns:
            Number of junction records created
        """
        created = 0
        for seq_num, call_id in enumerate(sorted_call_ids):
            # Check if junction already exists
            existing = (
                self._session.query(JudgedDatasetLLMCallORM)
                .filter_by(
                    judged_dataset_id=dataset_id,
                    llm_call_id=call_id,
                )
                .first()
            )

            if not existing:
                junction = JudgedDatasetLLMCallORM(
                    judged_dataset_id=dataset_id,
                    llm_call_id=call_id,
                    sequence_number=seq_num,
                )
                self._session.add(junction)
                created += 1

        return created
