"""SQL writer adapter for persisting LLM judgements to database.

Uses pure SQLAlchemy ORM models with deterministic UUIDs.
Auto-creates tables on first write and returns write summary for transparent logging.

Implements streaming pattern with write_one() for fault tolerance.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.infer.schemas.orms import (
    InferRunModel,
    LLMJudgementModel,
    InferWarningModel,
    WarningStage,
)
from llm_ensemble.infer.schemas.warnings import (
    PromptWarning,
    ProviderWarning,
    ParserWarning,
)
from llm_ensemble.infer.ports import JudgementWriter
from llm_ensemble.libs.db import (
    get_engine,
    create_all_tables,
    session_context,
)
from llm_ensemble.libs.db.uuid_helpers import (
    compute_llm_judgement_uuid,
    compute_infer_warning_uuid,
)
from llm_ensemble.libs.schemas.write_result import WriteResult


class SqlJudgementWriter(JudgementWriter):
    """SQL writer adapter for LLM judgements.

    Writes judgements to SQL database using pure SQLAlchemy ORM with streaming.

    Features:
    - Auto-creates tables on first write
    - Deterministic UUIDs for all entities (computed on-the-fly)
    - Central shared database across all runs for data accumulation
    - Idempotent writes via session.get() check (insert if new, skip if exists)
    - Streaming writes with write_one() for fault tolerance
    - Uses session_context() for transaction management per write
    - Returns WriteSummary for transparent logging (separation of concerns)

    Database URL is read from DATABASE_URL environment variable.
    Defaults to sqlite:///artifacts/llm_ensemble.db if not set.
    """

    def __init__(self, database_url: str | None = None):
        """Initialize SQL writer.

        Args:
            database_url: Database connection URL. If None, reads from DATABASE_URL env var
                         or defaults to sqlite:///artifacts/llm_ensemble.db
        """
        super().__init__()
        self.database_url = database_url
        self.engine: Engine = get_engine(database_url)
        self._session: Optional[Session] = None
        self._judgements_written = 0
        self._run_saved = False

    def open(self, run_dir: Path) -> "SqlJudgementWriter":
        """Initialize writer with run directory and prepare for streaming.

        Auto-creates tables on first write.

        Args:
            run_dir: Run directory (not used by SQL writer - writes to centralized database)

        Returns:
            Self, to enable context manager usage

        Raises:
            IOError: If writer cannot be initialized
        """
        # Auto-create tables
        create_all_tables(self.engine)
        return self

    def write_one(self, judgement: LLMJudgement) -> WriteResult:
        """Write a single judgement to the database.

        Judgement is persisted immediately for fault tolerance.
        Each write is a separate transaction.

        Args:
            judgement: LLMJudgement object to write

        Returns:
            WriteResult for this specific write operation

        Raises:
            IOError: If write operation fails
        """
        try:
            with session_context(self.engine) as session:
                # Save InferRun on first write (idempotent)
                if not self._run_saved:
                    self._save_infer_run(session, judgement)
                    self._run_saved = True

                # Compute judgement UUID
                judgement_id = compute_llm_judgement_uuid(
                    judgement.judging_sample.id,
                    judgement.run_info.id
                )

                # Check if judgement already exists (idempotent)
                existing = session.get(LLMJudgementModel, judgement_id)
                if existing:
                    # Skip duplicate
                    return WriteResult(item_id=judgement_id, item_type="judgement")

                # Create LLMJudgementModel (denormalized)
                judgement_model = LLMJudgementModel(
                    id=judgement_id,
                    judging_sample_id=judgement.judging_sample.id,
                    infer_run_id=judgement.run_info.id,
                    run_name=judgement.run_info.run_name,
                    # Request fields
                    prompt=judgement.llm_request.prompt,
                    # Response fields
                    raw_response=judgement.llm_response.raw_response,
                    latency_ms=judgement.llm_response.latency_ms,
                    retries=judgement.llm_response.retries,
                    cost_estimate_usd=judgement.llm_response.cost_estimate_usd,
                    # Score fields (nullable if parse failed)
                    label=judgement.llm_score.label.value if judgement.llm_score and judgement.llm_score.label else None,
                    confidence=judgement.llm_score.confidence if judgement.llm_score else None,
                    rationale=judgement.llm_score.rationale if judgement.llm_score else None,
                )
                session.add(judgement_model)

                # Save warnings from all stages
                self._save_warnings(session, judgement, judgement_id)

                self._judgements_written += 1

            return WriteResult(item_id=judgement_id, item_type="judgement")

        except Exception as e:
            raise IOError(f"Failed to write judgement to database: {e}") from e

    def close(self) -> WriteSummary:
        """Close writer and finalize output.

        Returns:
            WriteSummary tracking write operations performed during streaming

        Raises:
            IOError: If close operation fails
        """
        return WriteSummary(judgements_written=self._judgements_written)

    def _save_infer_run(self, session: Session, judgement: LLMJudgement) -> None:
        """Save infer run entity to database (idempotent).

        Args:
            session: SQLAlchemy session
            judgement: LLMJudgement containing run_info
        """
        run_info = judgement.run_info

        # Check if already exists
        existing = session.get(InferRunModel, run_info.id)
        if existing:
            return

        # Convert configs to dicts for JSONB storage
        infer_run_model = InferRunModel(
            id=run_info.id,
            run_name=run_info.run_name,
            run_type=run_info.run_type,
            model_config_name=run_info.model_config_name,
            prompt_config_name=run_info.prompt_config_name,
            io_config_name=run_info.io_config_name,
            model_config=run_info.model_cfg.model_dump(),
            prompt_config=run_info.prompt_config.model_dump(),
            io_config=run_info.io_config.model_dump(),
            input_file=run_info.input_file,
            limit=run_info.limit,
            git_sha=run_info.git_sha,
            git_branch=run_info.git_branch,
            git_is_dirty="true" if not run_info.git_clean else "false",
            notes=run_info.notes,
        )
        session.add(infer_run_model)

    def _save_warnings(
        self,
        session: Session,
        judgement: LLMJudgement,
        judgement_id
    ) -> None:
        """Save warnings from all pipeline stages.

        Args:
            session: SQLAlchemy session
            judgement: LLMJudgement containing warnings
            judgement_id: UUID of the judgement
        """
        # Collect warnings from all stages
        prompt_warnings = judgement.llm_request.warnings
        provider_warnings = judgement.llm_response.warnings
        parser_warnings = judgement.llm_score.warnings if judgement.llm_score else []

        # Save prompt warnings
        for warning in prompt_warnings:
            if isinstance(warning, PromptWarning):
                self._save_warning(
                    session,
                    judgement_id,
                    WarningStage.PROMPT,
                    warning.code.value,
                    warning.message,
                    warning.metadata,
                )

        # Save provider warnings
        for warning in provider_warnings:
            if isinstance(warning, ProviderWarning):
                self._save_warning(
                    session,
                    judgement_id,
                    WarningStage.PROVIDER,
                    warning.code.value,
                    warning.message,
                    warning.metadata,
                )

        # Save parser warnings
        for warning in parser_warnings:
            if isinstance(warning, ParserWarning):
                self._save_warning(
                    session,
                    judgement_id,
                    WarningStage.PARSER,
                    warning.code.value,
                    warning.message,
                    warning.metadata,
                )

    def _save_warning(
        self,
        session: Session,
        judgement_id,
        stage: WarningStage,
        code: str,
        message: str,
        metadata: dict,
    ) -> None:
        """Save a single warning to database (idempotent).

        Args:
            session: SQLAlchemy session
            judgement_id: UUID of parent judgement
            stage: Warning stage (PROMPT/PROVIDER/PARSER)
            code: Warning code
            message: Warning message
            metadata: Warning metadata dict
        """
        # Compute deterministic UUID
        warning_id = compute_infer_warning_uuid(
            judgement_id,
            stage.value,
            code,
            message,
        )

        # Check if already exists (idempotent)
        existing = session.get(InferWarningModel, warning_id)
        if existing:
            return

        warning_model = InferWarningModel(
            id=warning_id,
            judgement_id=judgement_id,
            stage=stage,
            code=code,
            message=message,
            metadata=metadata if metadata else None,
        )
        session.add(warning_model)
