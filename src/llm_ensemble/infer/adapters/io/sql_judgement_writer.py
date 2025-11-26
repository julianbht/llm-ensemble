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
"""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.db import (
    get_engine,
    get_session,
    compute_provider_uuid,
    compute_model_uuid,
    compute_model_config_uuid,
    compute_prompt_template_uuid,
    compute_parser_spec_uuid,
    compute_infer_run_uuid,
    compute_llm_response_text_uuid,
    compute_llm_invocation_metrics_uuid,
    compute_llm_score_uuid,
    compute_judged_dataset_fingerprint,
)
from llm_ensemble.infer.schemas.orms_normalized import (
    ProviderORM,
    ModelORM,
    ModelConfigORM,
    PromptTemplateORM,
    ParserSpecORM,
    InferRunORM,
    JudgedDatasetORM,
    DatasetJudgementORM,
    LLMPromptTextORM,
    LLMResponseTextORM,
    LLMInvocationMetricsORM,
    LLMScoreORM,
    LLMJudgementORM,
)
from llm_ensemble.infer.adapters.io.mappers_domain_to_orm import (
    provider_name_to_orm,
    model_config_to_model_orm,
    model_config_to_orm,
    prompt_config_to_template_orm,
    prompt_config_to_parser_orm,
    infer_run_info_to_orm,
    llm_prompt_to_orm,
    llm_response_text_to_orm,
    llm_invocation_metrics_to_orm,
    llm_score_to_orm,
    llm_judgement_to_orm,
    dataset_judgement_to_orm,
    judged_dataset_to_orm,
)
from llm_ensemble.libs.logging.log_events import InferWriteEvent


class SqlJudgementWriter(JudgementWriter):
    """Write LLMJudgement records to SQL database.

    Normalized schema: decomposes judgements into Provider, ModelConfig,
    PromptTemplate, ParserSpec, InferRun, LLMPromptText, LLMResponseText,
    LLMInvocationMetrics, LLMScore, DatasetJudgement, LLMJudgement entities.

    Deduplication via deterministic UUIDs + unique constraints.
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
        self._sequence_num: int = 0
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
    ) -> "SqlJudgementWriter":
        """Open database session and initialize run metadata."""
        if self._session is not None:
            raise RuntimeError("Writer is already open")

        engine = get_engine()
        self._session = get_session(engine)

        # Initialize run metadata (providers, models, prompts, etc.)
        self._initialize_run_metadata(run_info, normalized_dataset, start_idx, end_idx)

        # Create JudgedDataset with NULL fingerprint (finalized in close())
        self._judged_dataset_id = uuid.uuid4()
        judged_dataset_orm = JudgedDatasetORM(
            id=self._judged_dataset_id,
            fingerprint=None,
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
        llm_score_id = self._upsert_llm_score(judgement, llm_response_text_id) if judgement.llm_score else None

        # Create DatasetJudgement
        dataset_judgement_orm = dataset_judgement_to_orm(
            self._judged_dataset_id,
            self._sequence_num
        )
        self._session.add(dataset_judgement_orm)
        self._write_summary.add_judged_dataset_junctions(created=1)

        # Create LLMJudgement
        llm_judgement_orm = llm_judgement_to_orm(
            judgement,
            dataset_judgement_orm.id,
            self._model_config_id,
            llm_prompt_text_id,
            llm_invocation_metrics_id,
            llm_score_id,
        )
        self._session.add(llm_judgement_orm)
        self._write_summary.add_llm_judgements(created=1, skipped=0)

        self._sequence_num += 1
        self._session.commit()

    def close(self) -> WriteSummary:
        """Close session and finalize JudgedDataset."""
        if self._session is not None:
            # Finalize JudgedDataset fingerprint
            if self._judged_dataset_id and self._dataset_sample_ids:
                # Compute fingerprint from dataset_sample IDs (same as JudgedDataset.create())
                fingerprint = compute_judged_dataset_fingerprint(self._dataset_sample_ids)

                judged_dataset = self._session.get(JudgedDatasetORM, self._judged_dataset_id)
                if judged_dataset:
                    judged_dataset.fingerprint = fingerprint
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
    ) -> None:
        """Initialize shared metadata entities."""
        # Upsert Provider
        self._provider_id = self._upsert_entity(
            ProviderORM,
            compute_provider_uuid(run_info.model_cfg.provider),
            lambda: provider_name_to_orm(run_info.model_cfg.provider),
            "providers"
        )

        # Upsert Model
        self._model_id = self._upsert_entity(
            ModelORM,
            compute_model_uuid(run_info.model_cfg.model_id),
            lambda: model_config_to_model_orm(run_info.model_cfg),
            "models"
        )

        # Upsert ModelConfig
        self._model_config_id = self._upsert_entity(
            ModelConfigORM,
            compute_model_config_uuid(run_info.model_cfg.name),
            lambda: model_config_to_orm(run_info.model_cfg, self._model_id, self._provider_id),
            "model_configs"
        )

        # Upsert PromptTemplate
        builder = run_info.prompt_config.get_prompt_builder()
        template_text = getattr(builder, "template_text", "")
        self._prompt_template_id = self._upsert_entity(
            PromptTemplateORM,
            compute_prompt_template_uuid(run_info.prompt_config.name),
            lambda: prompt_config_to_template_orm(run_info.prompt_config, template_text),
            "prompt_templates"
        )

        # Upsert ParserSpec
        code_hash = "0" * 64  # Placeholder
        self._parser_spec_id = self._upsert_entity(
            ParserSpecORM,
            compute_parser_spec_uuid(
                run_info.prompt_config.parser_module,
                run_info.prompt_config.parser_class,
                code_hash
            ),
            lambda: prompt_config_to_parser_orm(run_info.prompt_config, code_hash),
            "parser_specs"
        )

        # Create InferRun
        infer_run_id = compute_infer_run_uuid(run_info.run_name)
        config_names = {
            "model_config": run_info.model_cfg.name,
            "prompt_template": run_info.prompt_config.name,
            "parser_spec": f"{run_info.prompt_config.parser_module}:{run_info.prompt_config.parser_class}",
        }
        infer_run_orm = infer_run_info_to_orm(
            run_info,
            config_names,
            start_idx,
            end_idx,
        )
        self._session.add(infer_run_orm)
        self._infer_run_id = infer_run_id
        self._write_summary.add_infer_runs(created=1, skipped=0)

        self._session.commit()

    def _upsert_entity(self, orm_class, entity_id: uuid.UUID, create_fn, entity_name: str) -> uuid.UUID:
        """Generic upsert helper."""
        existing = self._session.get(orm_class, entity_id)
        if existing:
            # Track skip
            attr_name = f"add_{entity_name}"
            if hasattr(self._write_summary, attr_name):
                getattr(self._write_summary, attr_name)(created=0, skipped=1)
            return entity_id

        entity_orm = create_fn()
        self._session.add(entity_orm)

        # Track create
        attr_name = f"add_{entity_name}"
        if hasattr(self._write_summary, attr_name):
            getattr(self._write_summary, attr_name)(created=1, skipped=0)

        return entity_id

    def _upsert_llm_prompt_text(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMPromptText from judgement."""
        llm_prompt_text_orm = llm_prompt_to_orm(
            judgement.llm_prompt,
            self._prompt_template_id,
            judgement.llm_prompt.dataset_sample.id,
        )
        existing = self._session.get(LLMPromptTextORM, llm_prompt_text_orm.id)
        if not existing:
            self._session.add(llm_prompt_text_orm)
            self._write_summary.add_llm_prompts(created=1, skipped=0)
        else:
            self._write_summary.add_llm_prompts(created=0, skipped=1)
        return llm_prompt_text_orm.id

    def _upsert_llm_response_text(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMResponseText from judgement."""
        response_text = judgement.llm_score.llm_response_text if judgement.llm_score else ""
        response_id = compute_llm_response_text_uuid(response_text)
        existing = self._session.get(LLMResponseTextORM, response_id)
        if not existing:
            response_orm = llm_response_text_to_orm(response_text)
            self._session.add(response_orm)
            self._write_summary.add_llm_responses(created=1, skipped=0)
        else:
            self._write_summary.add_llm_responses(created=0, skipped=1)
        return response_id

    def _upsert_llm_invocation_metrics(self, judgement: LLMJudgement) -> uuid.UUID:
        """Upsert LLMInvocationMetrics from judgement."""
        metrics_id = compute_llm_invocation_metrics_uuid(
            latency_ms=judgement.invocation_metrics.latency_ms,
            retries=judgement.invocation_metrics.retries,
            cost_estimate_usd=judgement.invocation_metrics.cost_estimate_usd,
            generation_id=judgement.invocation_metrics.generation_id,
            prompt_tokens=judgement.invocation_metrics.prompt_tokens,
            completion_tokens=judgement.invocation_metrics.completion_tokens,
            total_tokens=judgement.invocation_metrics.total_tokens,
        )
        existing = self._session.get(LLMInvocationMetricsORM, metrics_id)
        if not existing:
            metrics_orm = llm_invocation_metrics_to_orm(judgement.invocation_metrics)
            self._session.add(metrics_orm)
            self._write_summary.add_llm_invocation_metrics(created=1, skipped=0)
        else:
            self._write_summary.add_llm_invocation_metrics(created=0, skipped=1)
        return metrics_id

    def _upsert_llm_score(self, judgement: LLMJudgement, llm_response_text_id: uuid.UUID) -> uuid.UUID:
        """Upsert LLMScore from judgement."""
        score_id = compute_llm_score_uuid(self._parser_spec_id, llm_response_text_id)
        existing = self._session.get(LLMScoreORM, score_id)
        if not existing:
            score_orm = llm_score_to_orm(
                judgement.llm_score,
                self._parser_spec_id,
                llm_response_text_id,
            )
            self._session.add(score_orm)
            self._write_summary.add_llm_scores(created=1, skipped=0)
        else:
            self._write_summary.add_llm_scores(created=0, skipped=1)
        return score_id
