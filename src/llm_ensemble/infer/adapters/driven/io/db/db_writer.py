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
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.infer_run import InferRun
from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.infer.application.write_summary import WriteSummary
from llm_ensemble.infer.application.ports.driven.for_output import ForOutput
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
    infer_run_config_to_orm,
    infer_run_to_orm,
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
        infer_run: InferRun,
    ) -> DBWriter:
        """Open database session and initialize run metadata.

        Args:
            infer_run: InferRun aggregate root (config present, output=None)

        Returns:
            Self for method chaining
        """
        if self._session is not None:
            raise RuntimeError("Writer is already open")

        engine = get_engine()
        self._session = get_session(engine)

        # Upsert InferRunConfig components and create InferRunConfig
        # This initializes _infer_run_config_id for use in write_one()
        self._upsert_infer_run_config(infer_run.infer_run_config)

        # Create InferRunOutput FIRST (before InferRun) so judgements can reference it
        # Use same ID as InferRun (1:1 relationship)
        # Start with NULL fingerprint - will be updated in close()
        infer_run_output = InferRunOutput(
            id=infer_run.id,  # Same ID as InferRun (1:1 relationship)
            llm_judgements=[],  # Judgements written individually via write_one()
            sample_fingerprint=None,  # Set in close() after all samples collected
            judgement_count=0,  # Updated in close()
            error_count=0,  # Updated in close()
            avg_latency_ms=0.0,  # Updated in close()
        )
        infer_run_output_orm = infer_run_output_to_orm(infer_run_output)
        self._session.add(infer_run_output_orm)
        self._infer_run_output_id = infer_run.id
        self._write_summary.add_infer_run_outputs(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_INFER_RUN_OUTPUTS, created=1, skipped=0)

        # Create InferRun (always new - unique run_name constraint)
        # Link to InferRunOutput created above
        infer_run_orm = infer_run_to_orm(
            infer_run=infer_run,
            infer_run_config_id=self._infer_run_config_id,
            infer_run_output_id=self._infer_run_output_id,
        )
        self._session.add(infer_run_orm)
        self._infer_run_id = infer_run.id
        self._write_summary.add_infer_runs(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_INFER_RUNS, created=1, skipped=0)
        self._session.commit()

        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to database.

        LLM judgements are always unique per run (no deduplication needed).
        Each run produces new judgements even for the same samples.
        """
        if self._session is None:
            raise RuntimeError("Writer is not open")
        if self._infer_run_output_id is None:
            raise RuntimeError("InferRunOutput not initialized")

        # Track dataset_sample ID for fingerprint computation in close()
        self._dataset_sample_ids.append(judgement.dataset_sample.id)

        # Upsert LLMPromptText
        llm_prompt_text_id = self._upsert_llm_prompt_text(judgement)

        # Upsert LLMResponseText
        llm_response_text_id = self._upsert_llm_response_text(judgement)

        # Upsert LLMScore
        llm_score_id = self._upsert_llm_score(judgement)

        # Create LLMJudgement (always new - unique per run)
        # InferRunOutput was created in open(), so FK constraint is satisfied
        llm_judgement_orm = llm_judgement_to_orm(
            judgement,
            self._infer_run_output_id,
            judgement.dataset_sample.id,
            llm_prompt_text_id,
            llm_response_text_id,
            llm_score_id,
        )
        self._session.add(llm_judgement_orm)
        self._session.flush()
        self._write_summary.add_llm_judgements(created=1, skipped=0)

        self._session.commit()

    def close(self) -> WriteSummary:
        """Close session and finalize InferRunOutput and end_time."""
        if self._session is not None:
            # Capture end time
            end_time = datetime.now()

            # Update InferRunOutput with fingerprint (created in open())
            if self._dataset_sample_ids:
                fingerprint = compute_judged_dataset_fingerprint(self._dataset_sample_ids)

                # Update existing InferRunOutput with computed fingerprint
                infer_run_output_orm = self._session.get(InferRunOutputORM, self._infer_run_output_id)
                infer_run_output_orm.sample_fingerprint = fingerprint

                # Update InferRun end_time
                infer_run_orm = self._session.get(InferRunORM, self._infer_run_id)
                infer_run_orm.end_time = end_time

                self._session.commit()

            # Log per-judgement entity totals (accumulated during write_one calls)
            if self._write_summary.llm_prompts_created > 0 or self._write_summary.llm_prompts_skipped > 0:
                self.logger.info(
                    InferWriteEvent.WRITE_LLM_PROMPTS,
                    created=self._write_summary.llm_prompts_created,
                    skipped=self._write_summary.llm_prompts_skipped,
                )
            if self._write_summary.llm_responses_created > 0 or self._write_summary.llm_responses_skipped > 0:
                self.logger.info(
                    InferWriteEvent.WRITE_LLM_RESPONSES,
                    created=self._write_summary.llm_responses_created,
                    skipped=self._write_summary.llm_responses_skipped,
                )
            if self._write_summary.llm_scores_created > 0 or self._write_summary.llm_scores_skipped > 0:
                self.logger.info(
                    InferWriteEvent.WRITE_LLM_SCORES,
                    created=self._write_summary.llm_scores_created,
                    skipped=self._write_summary.llm_scores_skipped,
                )
            if self._write_summary.llm_judgements_created > 0 or self._write_summary.llm_judgements_skipped > 0:
                self.logger.info(
                    InferWriteEvent.WRITE_LLM_JUDGEMENTS,
                    created=self._write_summary.llm_judgements_created,
                    skipped=self._write_summary.llm_judgements_skipped,
                )

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
        Stores _infer_run_config_id (actual UUID from database) for use in write_one().

        IMPORTANT: Returns actual UUIDs from database to ensure correct FK references.
        """
        # Upsert Provider and get actual UUID
        created, skipped, provider_id = self._upsert_provider(infer_run_config.provider)
        self._write_summary.add_providers(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_PROVIDERS, created=created, skipped=skipped)

        # Upsert ModelConfig and get actual UUID
        created, skipped, model_config_id = self._upsert_model_config(infer_run_config.model_cfg)
        self._write_summary.add_model_configs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_MODEL_CONFIGS, created=created, skipped=skipped)

        # Upsert PromptBuilder and get actual UUID
        created, skipped, prompt_builder_id = self._upsert_prompt_builder(
            infer_run_config.prompt_template.prompt_builder
        )
        # Note: PromptBuilder tracking is done via PromptTemplate

        # Upsert Parser and get actual UUID
        created, skipped, parser_id = self._upsert_parser(
            infer_run_config.prompt_template.response_parser
        )
        self._write_summary.add_parser(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_PARSERS, created=created, skipped=skipped)

        # Upsert PromptTemplate (bundles prompt_builder + parser) and get actual UUID
        created, skipped, prompt_template_id = self._upsert_prompt_template(
            infer_run_config.prompt_template,
            prompt_builder_id,
            parser_id,
        )
        self._write_summary.add_prompt_templates(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_PROMPT_TEMPLATES, created=created, skipped=skipped)

        # Upsert InferRunConfig (bundles all components, execution context inlined) and get actual UUID
        created, skipped, infer_run_config_id = self._upsert_infer_run_config_entity(
            infer_run_config,
            provider_id,
            model_config_id,
            prompt_template_id,
        )
        self._write_summary.add_infer_run_configs(created=created, skipped=skipped)
        if created > 0 or skipped > 0:
            self.logger.info(InferWriteEvent.WRITE_INFER_RUN_CONFIGS, created=created, skipped=skipped)
        # CRITICAL: Use actual UUID from database, not random domain UUID
        self._infer_run_config_id = infer_run_config_id

    def _upsert_provider(self, provider) -> Tuple[int, int, uuid.UUID]:
        """Upsert Provider entity and return actual database UUID.

        Returns:
            Tuple of (created_count, skipped_count, actual_uuid)
        """
        provider_orm = provider_to_orm(provider)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(provider_orm)
            self._session.flush()
            return (1, 0, provider_orm.id)
        except IntegrityError:
            savepoint.rollback()
            # Query for existing provider by natural key
            existing = self._session.query(ProviderORM).filter_by(
                name=provider.name,
                version=provider.version,
            ).first()
            return (0, 1, existing.id)

    def _upsert_model_config(self, model_cfg) -> Tuple[int, int, uuid.UUID]:
        """Upsert ModelConfig entity and return actual database UUID.

        Returns:
            Tuple of (created_count, skipped_count, actual_uuid)
        """
        model_config_orm = model_config_to_orm(model_cfg)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(model_config_orm)
            self._session.flush()
            return (1, 0, model_config_orm.id)
        except IntegrityError:
            savepoint.rollback()
            # Query for existing model config by natural key
            existing = self._session.query(ModelConfigORM).filter_by(
                name=model_cfg.name,
            ).first()
            return (0, 1, existing.id)

    def _upsert_prompt_builder(self, prompt_builder) -> Tuple[int, int, uuid.UUID]:
        """Upsert PromptBuilder entity and return actual database UUID.

        Returns:
            Tuple of (created_count, skipped_count, actual_uuid)
        """
        prompt_builder_orm = prompt_builder_to_orm(prompt_builder)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(prompt_builder_orm)
            self._session.flush()
            return (1, 0, prompt_builder_orm.id)
        except IntegrityError:
            savepoint.rollback()
            # Query for existing prompt builder by natural key
            existing = self._session.query(PromptBuilderORM).filter_by(
                name=prompt_builder.name,
                version=prompt_builder.version,
            ).first()
            return (0, 1, existing.id)

    def _upsert_parser(self, parser) -> Tuple[int, int, uuid.UUID]:
        """Upsert Parser entity and return actual database UUID.

        Returns:
            Tuple of (created_count, skipped_count, actual_uuid)
        """
        parser_orm = parser_to_orm(parser)
        try:
            savepoint = self._session.begin_nested()
            self._session.add(parser_orm)
            self._session.flush()
            return (1, 0, parser_orm.id)
        except IntegrityError:
            savepoint.rollback()
            # Query for existing parser by natural key
            existing = self._session.query(ParserORM).filter_by(
                name=parser.name,
                version=parser.version,
            ).first()
            return (0, 1, existing.id)

    def _upsert_prompt_template(
        self,
        prompt_template,
        prompt_builder_id: uuid.UUID,
        parser_id: uuid.UUID,
    ) -> Tuple[int, int, uuid.UUID]:
        """Upsert PromptTemplate entity and return actual database UUID.

        Args:
            prompt_template: PromptTemplate domain object
            prompt_builder_id: Actual UUID of PromptBuilder in database
            parser_id: Actual UUID of Parser in database

        Returns:
            Tuple of (created_count, skipped_count, actual_uuid)
        """
        prompt_template_orm = prompt_template_to_orm(prompt_template)
        # Override with actual database UUIDs
        prompt_template_orm.prompt_builder_id = prompt_builder_id
        prompt_template_orm.parser_id = parser_id

        try:
            savepoint = self._session.begin_nested()
            self._session.add(prompt_template_orm)
            self._session.flush()
            return (1, 0, prompt_template_orm.id)
        except IntegrityError:
            savepoint.rollback()
            # Query for existing prompt template by natural key
            existing = self._session.query(PromptTemplateORM).filter_by(
                name=prompt_template.name,
            ).first()
            return (0, 1, existing.id)

    def _upsert_infer_run_config_entity(
        self,
        infer_run_config,
        provider_id: uuid.UUID,
        model_config_id: uuid.UUID,
        prompt_template_id: uuid.UUID,
    ) -> Tuple[int, int, uuid.UUID]:
        """Upsert InferRunConfig entity and return actual database UUID.

        Args:
            infer_run_config: InferRunConfig domain object
            provider_id: Actual UUID of Provider in database
            model_config_id: Actual UUID of ModelConfig in database
            prompt_template_id: Actual UUID of PromptTemplate in database

        Returns:
            Tuple of (created_count, skipped_count, actual_uuid)
        """
        infer_run_config_orm = infer_run_config_to_orm(infer_run_config)
        # Override with actual database UUIDs
        infer_run_config_orm.provider_id = provider_id
        infer_run_config_orm.model_config_id = model_config_id
        infer_run_config_orm.prompt_template_id = prompt_template_id

        try:
            savepoint = self._session.begin_nested()
            self._session.add(infer_run_config_orm)
            self._session.flush()
            return (1, 0, infer_run_config_orm.id)
        except IntegrityError:
            savepoint.rollback()
            # Query for existing infer run config by natural key
            # Natural key includes all the FK IDs plus execution context
            existing = self._session.query(InferRunConfigORM).filter_by(
                model_config_id=model_config_id,
                provider_id=provider_id,
                prompt_template_id=prompt_template_id,
                input_run_name=infer_run_config.input_run_name,
                start_idx=infer_run_config.start_idx,
                end_idx=infer_run_config.end_idx,
                io_name=infer_run_config.io_name,
            ).first()
            return (0, 1, existing.id)

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
