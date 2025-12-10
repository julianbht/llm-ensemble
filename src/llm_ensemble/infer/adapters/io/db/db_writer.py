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

from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import OutputPort
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.db import (
    get_engine,
    get_session,
    compute_judged_dataset_fingerprint,
)
from llm_ensemble.infer.adapters.io.db.orms import (
    ProviderORM,
    ModelConfigORM,
    PromptTemplateORM,
    ParserORM,
    InferRunORM,
    JudgedDatasetORM,
    LLMPromptORM,
    LLMResponseTextORM,
    LLMInvocationMetricsORM,
    LLMScoreORM,
)
from llm_ensemble.infer.adapters.io.db.mappers_to_orm import (
    provider_name_to_orm,
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


class DBWriter(OutputPort):
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
    ) -> "DBWriter":
        """Open database session and initialize run metadata."""
        if self._session is not None:
            raise RuntimeError("Writer is already open")

        engine = get_engine()
        self._session = get_session(engine)

        # Compute actual start_idx and end_idx from run_info
        start_idx = run_info.start_idx if run_info.start_idx is not None else 0
        end_idx = run_info.end_idx if run_info.end_idx is not None else len(normalized_dataset.samples)

        # Create InferRun (always new)
        infer_run_orm = infer_run_info_to_orm(
            run_info,
            start_idx,
            end_idx,
        )
        self._session.add(infer_run_orm)
        self._infer_run_id = run_info.id
        self._write_summary.add_infer_runs(created=1, skipped=0)
        self._session.commit()

        # JudgedDataset will be created on first write_one() call
        # (needs provider_id and model_config_id from judgement)
        self._judged_dataset_id = None

        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to database."""
        if self._session is None:
            raise RuntimeError("Writer is not open")

        # On first call, upsert shared metadata and create JudgedDataset
        if self._judged_dataset_id is None:
            self._initialize_judged_dataset(judgement)

        # Track dataset_sample ID for fingerprint computation in close()
        self._dataset_sample_ids.append(judgement.llm_prompt.dataset_sample.id)

        # Upsert LLMPromptText (includes prompt_template upsert)
        llm_prompt_text_id = self._upsert_llm_prompt_text(judgement)

        # Upsert LLMResponseText
        llm_response_text_id = self._upsert_llm_response_text(judgement)

        # Upsert LLMInvocationMetrics
        llm_invocation_metrics_id = self._upsert_llm_invocation_metrics(judgement)

        # Upsert LLMScore (if present, includes parser upsert)
        llm_score_id = self._upsert_llm_score(judgement, llm_response_text_id) if judgement.llm_score else None

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

    def _initialize_judged_dataset(self, judgement: LLMJudgement) -> None:
        """Initialize JudgedDataset on first judgement write.

        Extracts provider and model_config from judgement and upserts them.
        """
        # Upsert Provider from judgement
        provider_orm = provider_name_to_orm(judgement.llm_provider.name)
        self._provider_id = self._upsert_by_name(
            ProviderORM,
            judgement.llm_provider.name,
            provider_orm,
            "providers"
        )

        # Upsert ModelConfig from judgement
        model_config_orm = model_config_to_orm(judgement.model_config)
        self._model_config_id = self._upsert_by_name(
            ModelConfigORM,
            judgement.model_config.name_hint,
            model_config_orm,
            "model_configs"
        )

        # Create JudgedDataset with same ID as InferRun (1:1 relationship)
        self._judged_dataset_id = self._infer_run_id
        judged_dataset_orm = JudgedDatasetORM(
            id=self._judged_dataset_id,
            model_config_id=self._model_config_id,
            provider_id=self._provider_id,
            sample_fingerprint=None,  # Computed in close()
        )
        self._session.add(judged_dataset_orm)
        self._session.commit()

        self._write_summary.add_judged_datasets(created=1, skipped=0)
        self.logger.info(InferWriteEvent.WRITE_JUDGED_DATASETS, created=1, skipped=0)

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
        stmt = select(LLMPromptORM).where(
            LLMPromptORM.prompt_template_id == judgement.prompt_template_id,
            LLMPromptORM.dataset_sample_id == judgement.llm_prompt.dataset_sample.id,
            LLMPromptORM.prompt_text == judgement.llm_prompt.prompt_text
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
        metrics_orm = llm_invocation_metrics_to_orm(judgement.llm_invocation_metrics)
        
        # Query by natural key (all fields)
        stmt = select(LLMInvocationMetricsORM).where(
            LLMInvocationMetricsORM.latency_ms == judgement.llm_invocation_metrics.latency_ms,
            LLMInvocationMetricsORM.retries == judgement.llm_invocation_metrics.retries,
            LLMInvocationMetricsORM.cost_estimate_usd == judgement.llm_invocation_metrics.cost_estimate_usd,
            LLMInvocationMetricsORM.generation_id == judgement.llm_invocation_metrics.generation_id,
            LLMInvocationMetricsORM.prompt_tokens == judgement.llm_invocation_metrics.prompt_tokens,
            LLMInvocationMetricsORM.completion_tokens == judgement.llm_invocation_metrics.completion_tokens,
            LLMInvocationMetricsORM.total_tokens == judgement.llm_invocation_metrics.total_tokens
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
