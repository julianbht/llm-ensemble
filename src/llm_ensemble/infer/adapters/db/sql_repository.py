"""SQL database adapter for writing LLM judgements.

Writes LLMJudgement records to a SQL database using SQLAlchemy ORM.
Decomposes denormalized domain objects into normalized relational entities.

Uses data mapper pattern: domain service works with LLMJudgement objects,
SQL writer maps them to ORM entities. Mapper logic lives in mappers_domain_to_orm.py.

Architecture:
- Run metadata initialized once in open()
- Per-judgement entities created in write_one() using mappers
- Immediate commits for fault tolerance
- JudgedDataset finalized in close()
- Deduplication via natural key queries (not deterministic UUIDs)
"""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import select

from llm_ensemble.infer.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.db import (
    get_engine,
    get_session,
    compute_judged_dataset_fingerprint,
)
from llm_ensemble.infer.adapters.db.orms import (
    ProviderORM,
    ModelORM,
    ModelConfigORM,
    PromptTemplateORM,
    ParserORM,
    InferRunORM,
    JudgedDatasetORM,
    LLMPromptTextORM,
    LLMResponseTextORM,
    LLMInvocationMetricsORM,
    LLMScoreORM,
)
from llm_ensemble.infer.adapters.db.mappers_to_orm import (
    provider_name_to_orm,
    model_config_to_model_orm,
    model_config_to_orm,
    prompt_name_to_template_orm,
    parser_name_to_orm,
    infer_run_info_to_orm,
    llm_prompt_to_orm,
    llm_response_text_to_orm,
    llm_invocation_metrics_to_orm,
    llm_score_to_orm,
    llm_judgement_to_orm,
)
from llm_ensemble.libs.logging.log_events import InferWriteEvent


class SqlJudgementWriter(JudgementWriter):
    """Write LLMJudgement records to SQL database.

    Normalized schema: decomposes judgements into Provider, ModelConfig,
    PromptTemplate, ParserSpec, InferRun, LLMPromptText, LLMResponseText,
    LLMInvocationMetrics, LLMScore, DatasetJudgement, LLMJudgement entities.

    Deduplication via natural key queries + unique constraints.
    """

    def __init__(self):
        super().__init__()
        self._session: Optional[Session] = None
        self._provider_id: Optional[uuid.UUID] = None
        self._model_id: Optional[uuid.UUID] = None
        self._model_config_id: Optional[uuid.UUID] = None
        self._prompt_template_id: Optional[uuid.UUID] = None
        self._parser_spec_id: Optional[uuid.UUID] = None
        self._infer_run_id: Optional[uuid.UUID] = None
        self._judged_dataset_id: Optional[uuid.UUID] = None
        self._dataset_sample_ids: list[uuid.UUID] = []
        self._write_summary = WriteSummary()
        self.logger = get_logger(component="sql_judgement_writer")

    def open(
        self,
        run_dir: Path,
        run_info: InferRunInfo,
        normalized_dataset: NormalizedDataset,
        start_idx: int,
        end_idx: int,
        prompt_name: str,
        parser_name: str,
        template_text: str,
    ) -> "SqlJudgementWriter":
        """Open database session and initialize run metadata."""
        if self._session is not None:
            raise RuntimeError("Writer is already open")

        engine = get_engine()
        self._session = get_session(engine)

        # Initialize run metadata (providers, models, prompts, etc.)
        self._initialize_run_metadata(
            run_info, normalized_dataset, start_idx, end_idx,
            prompt_name, parser_name, template_text
        )

        # Create JudgedDataset with same ID as InferRun (1:1 relationship)
        # sample_fingerprint computed in close() after all judgements written
        self._judged_dataset_id = self._infer_run_id
        judged_dataset_orm = JudgedDatasetORM(
            id=self._judged_dataset_id,
            model_config_id=self._model_config_id,
            sample_fingerprint=None,
        )
        self._session.add(judged_dataset_orm)
        self._session.commit()

        self._write_summary.add_judged_datasets(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_JUDGED_DATASETS, created=1, skipped=0)

        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to database."""
        if self._session is None:
            raise RuntimeError("Writer is not open")

        # Track dataset_sample ID for fingerprint computation in close()
        self._dataset_sample_ids.append(judgement.llm_prompt.dataset_sample.id)

        # Upsert LLMPromptText
        llm_prompt_text_id = self._upsert_llm_prompt_text(judgement)

        # Upsert LLMResponseText
        llm_response_text_id = self._upsert_llm_response_text(judgement)

        # Upsert LLMInvocationMetrics
        llm_invocation_metrics_id = self._upsert_llm_invocation_metrics(judgement)

        # Upsert LLMScore (if present)
        llm_score_id = self._upsert_llm_score(judgement, llm_response_text_id, self._parser_spec_id) if judgement.llm_score else None

        # Create LLMJudgement (directly linked to JudgedDataset)
        llm_judgement_orm = llm_judgement_to_orm(
            judgement,
            self._judged_dataset_id,
            llm_prompt_text_id,
            llm_invocation_metrics_id,
            llm_score_id,
        )
        self._session.add(llm_judgement_orm)
        self._write_summary.add_llm_judgements(created=1, skipped=0)

        self._session.commit()

    def close(self) -> WriteSummary:
        """Close session and finalize JudgedDataset."""
        if self._session is not None:
            # Finalize JudgedDataset sample_fingerprint and link to InferRun
            if self._judged_dataset_id and self._dataset_sample_ids:
                # Compute fingerprint from dataset_sample IDs (same as JudgedDataset.create())
                fingerprint = compute_judged_dataset_fingerprint(self._dataset_sample_ids)

                judged_dataset = self._session.get(JudgedDatasetORM, self._judged_dataset_id)
                judged_dataset.sample_fingerprint = fingerprint

                # Link InferRun to JudgedDataset (enables aggregate CLI to find judgements)
                infer_run = self._session.get(InferRunORM, self._infer_run_id)
                infer_run.judged_dataset_id = self._judged_dataset_id

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

    def _initialize_run_metadata(
        self,
        run_info: InferRunInfo,
        normalized_dataset: NormalizedDataset,
        start_idx: int,
        end_idx: int,
        prompt_name: str,
        parser_name: str,
        template_text: str,
    ) -> None:
        """Initialize shared metadata entities."""
        # Upsert Provider by name
        provider_orm = provider_name_to_orm(run_info.model_cfg.provider)
        self._provider_id = self._upsert_by_name(
            ProviderORM,
            run_info.model_cfg.provider,
            provider_orm,
            "providers"
        )

        # Upsert Model by name
        model_orm = model_config_to_model_orm(run_info.model_cfg)
        self._model_id = self._upsert_by_name(
            ModelORM,
            run_info.model_cfg.model_name,
            model_orm,
            "models"
        )

        # Upsert ModelConfig by name
        model_config_orm = model_config_to_orm(run_info.model_cfg, self._model_id, self._provider_id)
        self._model_config_id = self._upsert_by_name(
            ModelConfigORM,
            run_info.model_cfg.name,
            model_config_orm,
            "model_configs"
        )

        # Upsert PromptTemplate by name
        prompt_template_orm = prompt_name_to_template_orm(prompt_name, template_text)
        self._prompt_template_id = self._upsert_by_name(
            PromptTemplateORM,
            prompt_name,
            prompt_template_orm,
            "prompt_templates"
        )

        # Upsert ParserSpec by name
        parser_spec_orm = parser_name_to_orm(parser_name)
        self._parser_spec_id = self._upsert_by_name(
            ParserORM,
            parser_name,
            parser_spec_orm,
            "parser"
        )

        # Create InferRun (always new)
        config_names = {
            "model_config": run_info.model_cfg.name,
            "prompt_name": prompt_name,
            "parser_name": parser_name,
        }
        infer_run_orm = infer_run_info_to_orm(
            run_info,
            config_names,
            start_idx,
            end_idx,
        )
        self._session.add(infer_run_orm)
        self._infer_run_id = run_info.id
        self._write_summary.add_infer_runs(created=1, skipped=0)

        self._session.commit()

    def _upsert_by_name(self, orm_class, name: str, new_entity_orm, entity_name: str) -> uuid.UUID:
        """Upsert entity by name (natural key for simple config entities).
        
        Args:
            orm_class: ORM class to query
            name: Natural key value
            new_entity_orm: New entity instance with random UUID
            entity_name: Entity name for metrics tracking
            
        Returns:
            UUID of existing or newly created entity
        """
        # Query by natural key
        stmt = select(orm_class).where(orm_class.name == name)
        existing = self._session.execute(stmt).scalar_one_or_none()
        
        if existing:
            # Track skip
            attr_name = f"add_{entity_name}"
            if hasattr(self._write_summary, attr_name):
                getattr(self._write_summary, attr_name)(created=0, skipped=1)
            return existing.id

        # Create new
        self._session.add(new_entity_orm)
        
        # Track create
        attr_name = f"add_{entity_name}"
        if hasattr(self._write_summary, attr_name):
            getattr(self._write_summary, attr_name)(created=1, skipped=0)

        return new_entity_orm.id

    def _upsert_llm_prompt_text(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMPromptText by natural key (prompt_template_id, dataset_sample_id, prompt_text)."""
        llm_prompt_text_orm = llm_prompt_to_orm(
            judgement.llm_prompt,
            judgement.prompt_template_id,
            judgement.llm_prompt.dataset_sample.id,
        )
        
        # Query by natural key
        stmt = select(LLMPromptTextORM).where(
            LLMPromptTextORM.prompt_template_id == judgement.prompt_template_id,
            LLMPromptTextORM.dataset_sample_id == judgement.llm_prompt.dataset_sample.id,
            LLMPromptTextORM.prompt_text == judgement.llm_prompt.prompt_text
        )
        existing = self._session.execute(stmt).scalar_one_or_none()
        
        if existing:
            self._write_summary.add_llm_prompts(created=0, skipped=1)
            self.logger.info(InferWriteEvent.WRITE_LLM_PROMPTS, created=0, skipped=1)
            return existing.id
        
        self._session.add(llm_prompt_text_orm)
        self._write_summary.add_llm_prompts(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_LLM_PROMPTS, created=1, skipped=0)
        return llm_prompt_text_orm.id

    def _upsert_llm_response_text(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMResponseText by natural key (llm_response_text)."""
        response_text = judgement.llm_score.llm_response_text if judgement.llm_score else ""
        response_orm = llm_response_text_to_orm(response_text)
        
        # Query by natural key
        stmt = select(LLMResponseTextORM).where(
            LLMResponseTextORM.llm_response_text == response_text
        )
        existing = self._session.execute(stmt).scalar_one_or_none()
        
        if existing:
            self._write_summary.add_llm_responses(created=0, skipped=1)
            self.logger.info(InferWriteEvent.WRITE_LLM_RESPONSES, created=0, skipped=1)
            return existing.id
        
        self._session.add(response_orm)
        self._write_summary.add_llm_responses(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_LLM_RESPONSES, created=1, skipped=0)
        return response_orm.id

    def _upsert_llm_invocation_metrics(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMInvocationMetrics by natural key (all metric fields)."""
        metrics_orm = llm_invocation_metrics_to_orm(judgement.invocation_metrics)
        
        # Query by natural key (all fields)
        stmt = select(LLMInvocationMetricsORM).where(
            LLMInvocationMetricsORM.latency_ms == judgement.invocation_metrics.latency_ms,
            LLMInvocationMetricsORM.retries == judgement.invocation_metrics.retries,
            LLMInvocationMetricsORM.cost_estimate_usd == judgement.invocation_metrics.cost_estimate_usd,
            LLMInvocationMetricsORM.generation_id == judgement.invocation_metrics.generation_id,
            LLMInvocationMetricsORM.prompt_tokens == judgement.invocation_metrics.prompt_tokens,
            LLMInvocationMetricsORM.completion_tokens == judgement.invocation_metrics.completion_tokens,
            LLMInvocationMetricsORM.total_tokens == judgement.invocation_metrics.total_tokens
        )
        existing = self._session.execute(stmt).scalar_one_or_none()
        
        if existing:
            self._write_summary.add_llm_invocation_metrics(created=0, skipped=1)
            self.logger.info(InferWriteEvent.WRITE_LLM_INVOCATION_METRICS, created=0, skipped=1)
            return existing.id
        
        self._session.add(metrics_orm)
        self._write_summary.add_llm_invocation_metrics(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_LLM_INVOCATION_METRICS, created=1, skipped=0)
        return metrics_orm.id

    def _upsert_llm_score(self, judgement: LLMJudgement, llm_response_text_id: uuid.UUID, parser_spec_id: uuid.UUID) -> uuid.UUID:
        """Upsert LLMScore by natural key (parser_spec_id, llm_response_text_id)."""
        score_orm = llm_score_to_orm(
            judgement.llm_score,
            parser_spec_id,
            llm_response_text_id,
        )
        
        # Query by natural key
        stmt = select(LLMScoreORM).where(
            LLMScoreORM.parser_spec_id == parser_spec_id,
            LLMScoreORM.llm_response_text_id == llm_response_text_id
        )
        existing = self._session.execute(stmt).scalar_one_or_none()
        
        if existing:
            self._write_summary.add_llm_scores(created=0, skipped=1)
            self.logger.info(InferWriteEvent.WRITE_LLM_SCORES, created=0, skipped=1)
            return existing.id
        
        self._session.add(score_orm)
        self._write_summary.add_llm_scores(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_LLM_SCORES, created=1, skipped=0)
        return score_orm.id
