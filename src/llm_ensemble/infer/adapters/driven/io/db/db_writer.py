"""SQL database adapter for writing LLM judgements.

Writes LLMJudgement records to a SQL database using SQLAlchemy ORM.
Decomposes denormalized domain objects into normalized relational entities.

Uses constraint-based duplicate detection via IntegrityError (same pattern as ingest CLI).

Architecture:
- Run metadata initialized once in open()
- Per-judgement entities created in write_one() using mappers
- Immediate commits for fault tolerance
- InferRunOutput finalized in close()
- Deduplication via database constraints + savepoints (not SELECT queries)
"""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.application.write_summary import WriteSummary
from llm_ensemble.infer.application.ports.driven.for_output import ForOutput
from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.db import (
    get_engine,
    get_session,
    compute_judged_dataset_fingerprint,
)
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    ProviderORM,
    ModelConfigORM,
    PromptBuilderORM,
    ParserORM,
    PromptTemplateORM,
    IngestRunContextORM,
    InferRunConfigORM,
    InferRunORM,
    InferRunOutputORM,
    LLMPromptTextORM,
    LLMResponseTextORM,
    LLMScoreORM,
    LLMJudgementORM,
)
from llm_ensemble.infer.adapters.driven.io.db.mappers_to_orm import (
    provider_to_orm,
    model_config_to_orm,
    prompt_builder_to_orm,
    parser_to_orm,
    prompt_template_to_orm,
    ingest_run_context_to_orm,
    infer_run_config_to_orm,
    infer_run_info_to_orm,
    infer_run_output_to_orm,
    llm_prompt_text_to_orm,
    llm_response_text_to_orm,
    llm_score_to_orm,
    llm_judgement_to_orm,
)
from llm_ensemble.libs.logging.log_events import InferWriteEvent


class DBWriter(ForOutput):
    """Write LLMJudgement records to SQL database.

    Normalized schema: decomposes judgements into Provider, ModelConfig,
    PromptBuilder, Parser, PromptTemplate, IngestRunContext, InferRunConfig,
    InferRun, InferRunOutput, LLMPromptText, LLMResponseText, LLMScore, LLMJudgement.

    Deduplication via database constraints + IntegrityError (same as ingest CLI).
    """

    def __init__(self, io_name: str):
        """Initialize writer with IO format name.

        Args:
            io_name: Name of the IO format (e.g., 'db_to_json')
        """
        self._io_name: str = io_name
        self._write_summary: Optional[WriteSummary] = None
        self._session: Optional[Session] = None
        self._infer_run_id: Optional[uuid.UUID] = None
        self._infer_run_config_id: Optional[uuid.UUID] = None
        self._infer_run_output_id: Optional[uuid.UUID] = None
        self._dataset_sample_ids: list[uuid.UUID] = []
        self._write_summary = WriteSummary()
        self.logger = get_logger(component="db_writer")

    @property
    def io_name(self) -> str:
        """Get I/O adapter name."""
        return self._io_name

    def open(
        self,
        run_info: InferRunInfo,
        infer_run_config: InferRunConfig,
    ) -> DBWriter:
        """Open database session and initialize run metadata.

        Args:
            run_info: Run metadata (git info, timestamps)
            infer_run_config: Complete configuration bundle with resolved indices

        Returns:
            Self for method chaining
        """
        if self._session is not None:
            raise RuntimeError("Writer is already open")

        engine = get_engine()
        self._session = get_session(engine)

        # Get resolved indices from infer_run_config
        context = infer_run_config.ingest_run_context
        start_idx = context.start_idx
        end_idx = context.end_idx

        # Create InferRun (always new - unique run_name constraint)
        infer_run_orm = infer_run_info_to_orm(run_info, start_idx, end_idx)
        self._session.add(infer_run_orm)
        self._infer_run_id = run_info.id
        self._write_summary.add_infer_runs(created=1, skipped=0)
        self._session.commit()

        # Upsert InferRunConfig components and create InferRunConfig
        # This initializes _infer_run_config_id for use in write_one()
        self._upsert_infer_run_config(infer_run_config)

        # InferRunOutput will be created in close() after all judgements written
        self._infer_run_output_id = None

        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to database.

        Uses savepoint + IntegrityError pattern for deduplication.
        """
        if self._session is None:
            raise RuntimeError("Writer is not open")
        if self._infer_run_config_id is None:
            raise RuntimeError("InferRunConfig not initialized")

        # Track dataset_sample ID for fingerprint computation in close()
        self._dataset_sample_ids.append(judgement.dataset_sample.id)

        # Upsert LLMPromptText
        llm_prompt_text_id = self._upsert_llm_prompt_text(judgement)

        # Upsert LLMResponseText
        llm_response_text_id = self._upsert_llm_response_text(judgement)

        # Upsert LLMScore
        llm_score_id = self._upsert_llm_score(judgement)

        # Create LLMJudgement (unique constraint on infer_run_output_id + dataset_sample_id)
        # Note: We'll create InferRunOutput in close(), so use infer_run_id as placeholder
        llm_judgement_orm = llm_judgement_to_orm(
            judgement,
            self._infer_run_id,  # Temporary - will be updated to infer_run_output_id in close()
            judgement.dataset_sample.id,
            llm_prompt_text_id,
            llm_response_text_id,
            llm_score_id,
        )
        try:
            savepoint = self._session.begin_nested()
            self._session.add(llm_judgement_orm)
            self._session.flush()
            self._write_summary.add_llm_judgements(created=1, skipped=0)
        except IntegrityError:
            savepoint.rollback()
            self._write_summary.add_llm_judgements(created=0, skipped=1)

        self._session.commit()

    def close(self) -> WriteSummary:
        """Close session and finalize InferRunOutput."""
        if self._session is not None:
            # Create InferRunOutput with fingerprint
            if self._dataset_sample_ids:
                fingerprint = compute_judged_dataset_fingerprint(self._dataset_sample_ids)

                infer_run_output_orm = infer_run_output_to_orm(
                    self._infer_run_id,  # Use same ID as InferRun (1:1 relationship)
                    self._infer_run_config_id,
                    fingerprint,
                )
                try:
                    savepoint = self._session.begin_nested()
                    self._session.add(infer_run_output_orm)
                    self._session.flush()
                    self._infer_run_output_id = self._infer_run_id
                    self._write_summary.add_infer_run_outputs(created=1, skipped=0)
                except IntegrityError:
                    savepoint.rollback()
                    self._write_summary.add_infer_run_outputs(created=0, skipped=1)

                # Link InferRun to InferRunOutput
                infer_run = self._session.get(InferRunORM, self._infer_run_id)
                infer_run.infer_run_output_id = self._infer_run_output_id

                self._session.commit()

            # Log totals
            self.logger.info(
                InferWriteEvent.WRITE_COMPLETE,
                total_created=self._write_summary.total_created,
                total_skipped=self._write_summary.total_skipped,
            )

            self._session.close()
            self._session = None

        summary = self._write_summary
        self._write_summary = WriteSummary()
        return summary

    def _upsert_infer_run_config(self, infer_run_config: InferRunConfig) -> None:
        """Upsert InferRunConfig and all its components.

        Uses savepoint + IntegrityError pattern for each entity.
        Stores _infer_run_config_id for use in write_one().
        """
        # Upsert Provider
        created, skipped = self._upsert_provider(infer_run_config.provider)
        self._write_summary.add_providers(created=created, skipped=skipped)

        # Upsert ModelConfig
        created, skipped = self._upsert_model_config(infer_run_config.model_cfg)
        self._write_summary.add_model_configs(created=created, skipped=skipped)

        # Upsert PromptBuilder (includes template_text)
        created, skipped = self._upsert_prompt_builder(infer_run_config.prompt_template.prompt_builder)
        self._write_summary.add_prompt_builders(created=created, skipped=skipped)

        # Upsert Parser
        created, skipped = self._upsert_parser(infer_run_config.prompt_template.response_parser)
        self._write_summary.add_parsers(created=created, skipped=skipped)

        # Upsert PromptTemplate (bundles prompt_builder + parser)
        created, skipped = self._upsert_prompt_template(infer_run_config.prompt_template)
        self._write_summary.add_prompt_templates(created=created, skipped=skipped)

        # Upsert IngestRunContext
        created, skipped = self._upsert_ingest_run_context(infer_run_config.ingest_run_context)
        self._write_summary.add_ingest_run_contexts(created=created, skipped=skipped)

        # Upsert InferRunConfig (bundles all components)
        created, skipped = self._upsert_infer_run_config_entity(infer_run_config)
        self._write_summary.add_infer_run_configs(created=created, skipped=skipped)
        self._infer_run_config_id = infer_run_config.id

    def _upsert_provider(self, provider) -> Tuple[int, int]:
        """Upsert Provider entity."""
        provider_orm = provider_to_orm(provider)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(provider_orm)
            self._session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _upsert_model_config(self, model_cfg) -> Tuple[int, int]:
        """Upsert ModelConfig entity."""
        model_config_orm = model_config_to_orm(model_cfg)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(model_config_orm)
            self._session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _upsert_prompt_builder(self, prompt_builder) -> Tuple[int, int]:
        """Upsert PromptBuilder entity."""
        prompt_builder_orm = prompt_builder_to_orm(prompt_builder)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(prompt_builder_orm)
            self._session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _upsert_parser(self, parser) -> Tuple[int, int]:
        """Upsert Parser entity."""
        parser_orm = parser_to_orm(parser)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(parser_orm)
            self._session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _upsert_prompt_template(self, prompt_template) -> Tuple[int, int]:
        """Upsert PromptTemplate entity."""
        prompt_template_orm = prompt_template_to_orm(prompt_template)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(prompt_template_orm)
            self._session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _upsert_ingest_run_context(self, ingest_run_context) -> Tuple[int, int]:
        """Upsert IngestRunContext entity."""
        ingest_run_context_orm = ingest_run_context_to_orm(ingest_run_context)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(ingest_run_context_orm)
            self._session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _upsert_infer_run_config_entity(self, infer_run_config) -> Tuple[int, int]:
        """Upsert InferRunConfig entity."""
        infer_run_config_orm = infer_run_config_to_orm(infer_run_config)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(infer_run_config_orm)
            self._session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _upsert_llm_prompt_text(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMPromptText and return its ID."""
        prompt_text_id = uuid.uuid4()
        llm_prompt_text_orm = llm_prompt_text_to_orm(
            judgement.prompt_text,
            prompt_text_id,
        )
        try:
            savepoint = self._session.begin_nested()
            self._session.add(llm_prompt_text_orm)
            self._session.flush()
            self._write_summary.add_llm_prompts(created=1, skipped=0)
            return prompt_text_id
        except IntegrityError:
            savepoint.rollback()
            self._write_summary.add_llm_prompts(created=0, skipped=1)
            # On duplicate, query to get existing ID
            stmt = self._session.query(LLMPromptTextORM).filter_by(
                prompt_text=judgement.prompt_text
            )
            existing = stmt.first()
            return existing.id

    def _upsert_llm_response_text(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMResponseText and return its ID."""
        response_text = judgement.response_text
        response_text_id = uuid.uuid4()
        response_text_orm = llm_response_text_to_orm(response_text, response_text_id)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(response_text_orm)
            self._session.flush()
            self._write_summary.add_llm_responses(created=1, skipped=0)
            return response_text_id
        except IntegrityError:
            savepoint.rollback()
            self._write_summary.add_llm_responses(created=0, skipped=1)
            # On duplicate, query to get existing ID
            stmt = self._session.query(LLMResponseTextORM).filter_by(
                llm_response_text=response_text
            )
            existing = stmt.first()
            return existing.id

    def _upsert_llm_score(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMScore and return its ID."""
        llm_score_orm = llm_score_to_orm(judgement.llm_score)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(llm_score_orm)
            self._session.flush()
            self._write_summary.add_llm_scores(created=1, skipped=0)
            return llm_score_orm.id
        except IntegrityError:
            savepoint.rollback()
            self._write_summary.add_llm_scores(created=0, skipped=1)
            # On duplicate, query to get existing ID by natural key
            stmt = self._session.query(LLMScoreORM).filter_by(
                label=judgement.llm_score.label,
                confidence=judgement.llm_score.confidence,
                rationale=judgement.llm_score.rationale,
            )
            existing = stmt.first()
            return existing.id
