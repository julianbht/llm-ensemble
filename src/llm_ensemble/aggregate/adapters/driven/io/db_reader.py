"""Database adapter for reading InferRunOutputs from infer runs.

Reads InferRunOutput records from PostgreSQL database by infer run name(s).
This adapter queries the infer schema and loads complete InferRunOutput domain
objects for use in the aggregation pipeline.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import joinedload

from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.llm_prompt_text import LLMPromptText
from llm_ensemble.infer.domain.entities.llm_response_text import LLMResponseText
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssue, ParserIssueCode
from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document

from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    LLMJudgementORM,
)
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    NormalizedDatasetJudgingSampleORM,
    JudgingSampleORM,
    QueryORM,
    DocumentORM,
)
from llm_ensemble.aggregate.application.ports.driven.for_input import ForInput
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class DBReader(ForInput):
    """Read InferRunOutput records from database by infer run name(s).

    This adapter implements the ForInput port by loading InferRunOutputs
    from the infer schema. It queries via InferRun → InferRunOutput relationship
    and reconstructs complete domain objects.

    Database connection:
    - Reads DATABASE_URL from environment (.env file)
    - Uses SQLAlchemy session with eager loading
    - Session closed automatically after read

    Query strategy:
    - For each run_name, find InferRunORM
    - Load linked InferRunOutputORM via FK
    - Query LLMJudgementORM records with eager loading of related entities:
      - dataset_sample (via NormalizedDatasetJudgingSampleORM)
      - judging_sample (via JudgingSampleORM)
      - query and document (via QueryORM, DocumentORM)
      - llm_prompt_text, llm_response_text, llm_score
    - Reconstruct Pydantic InferRunOutput models

    Note: This reader does NOT validate fingerprints or completeness.
    Validation belongs in the aggregation service layer.
    """

    def read(self, run_names: list[str]) -> list[InferRunOutput]:
        """Read InferRunOutput from database by infer run name(s).

        Loads one InferRunOutput per run, each containing the sample_fingerprint
        and all LLM judgements from that run.

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])

        Returns:
            List of InferRunOutput objects, one per run

        Raises:
            LookupError: If any infer run doesn't exist in database
        """
        engine = get_engine()

        with session_context(engine) as session:
            infer_run_outputs = []

            for run_name in run_names:
                # Find infer run by name with eager loading
                infer_run = (
                    session.query(InferRunORM)
                    .filter_by(run_name=run_name)
                    .options(joinedload(InferRunORM.infer_run_output))
                    .one_or_none()
                )

                if not infer_run:
                    raise LookupError(
                        f"Infer run '{run_name}' not found in database. "
                        f"Available runs: SELECT run_name FROM infer.infer_runs"
                    )

                infer_run_output_orm = infer_run.infer_run_output

                if not infer_run_output_orm:
                    raise LookupError(
                        f"Infer run '{run_name}' has no linked InferRunOutput. "
                        f"This indicates the run did not complete successfully."
                    )

                # Query all judgements for this infer_run_output with eager loading
                llm_judgement_orms = (
                    session.query(LLMJudgementORM)
                    .filter_by(infer_run_output_id=infer_run_output_orm.id)
                    .options(
                        joinedload(LLMJudgementORM.llm_prompt_text),
                        joinedload(LLMJudgementORM.llm_response_text),
                        joinedload(LLMJudgementORM.llm_score),
                    )
                    .all()
                )

                # Load dataset_sample data for all judgements
                # Need to query NormalizedDatasetJudgingSampleORM and join to JudgingSampleORM
                dataset_sample_ids = [j.dataset_sample_id for j in llm_judgement_orms]
                dataset_samples_orms = (
                    session.query(NormalizedDatasetJudgingSampleORM, JudgingSampleORM, QueryORM, DocumentORM)
                    .join(JudgingSampleORM, NormalizedDatasetJudgingSampleORM.judging_sample_id == JudgingSampleORM.id)
                    .join(QueryORM, JudgingSampleORM.query_id == QueryORM.id)
                    .join(DocumentORM, JudgingSampleORM.document_id == DocumentORM.id)
                    .filter(NormalizedDatasetJudgingSampleORM.id.in_(dataset_sample_ids))
                    .all()
                )

                # Build mapping of dataset_sample_id -> DatasetSample domain object
                dataset_sample_map: dict[UUID, NormalizedDatasetJudgingSample] = {}
                for ds_orm, js_orm, query_orm, doc_orm in dataset_samples_orms:
                    # Build Query domain object
                    query = Query(
                        id=query_orm.id,
                        query_text=query_orm.query_text,
                        content_hash=query_orm.content_hash,
                    )

                    # Build Document domain object
                    document = Document(
                        id=doc_orm.id,
                        doc_text=doc_orm.doc_text,
                        content_hash=doc_orm.content_hash,
                    )

                    # Build JudgingSample domain object
                    judging_sample = JudgingSample(
                        id=js_orm.id,
                        query=query,
                        document=document,
                        gold_score=RelevanceScore(js_orm.gold_score),
                    )

                    # Build DatasetSample domain object
                    dataset_sample = NormalizedDatasetJudgingSample(
                        id=ds_orm.id,
                        normalized_dataset_id=ds_orm.normalized_dataset_id,
                        judging_sample=judging_sample,
                        sequence_number=ds_orm.sequence_number,
                    )

                    dataset_sample_map[ds_orm.id] = dataset_sample

                # Convert LLMJudgementORMs to domain objects
                llm_judgements: list[LLMJudgement] = []
                for j_orm in llm_judgement_orms:
                    # Get dataset_sample from map
                    dataset_sample = dataset_sample_map[j_orm.dataset_sample_id]

                    # Build LLMInvocationMetrics
                    invocation_metrics = LLMInvocationMetrics(
                        latency_ms=j_orm.latency_ms,
                        retries=j_orm.retries,
                        cost_estimate_usd=j_orm.cost_estimate_usd,
                        actual_cost_usd=j_orm.actual_cost_usd,
                        generation_id=j_orm.generation_id,
                        prompt_tokens=j_orm.prompt_tokens,
                        completion_tokens=j_orm.completion_tokens,
                        total_tokens=j_orm.total_tokens,
                    )

                    # Build LLMScore (may be None if parsing failed)
                    llm_score = None
                    if j_orm.llm_score:
                        llm_score = LLMScore(
                            label=j_orm.llm_score.label,
                            confidence=j_orm.llm_score.confidence,
                            rationale=j_orm.llm_score.rationale,
                        )

                    # Reconstruct parser issue if present
                    parser_issue = None
                    if j_orm.parser_issue_code is not None:
                        parser_issue = ParserIssue(
                            code=ParserIssueCode(j_orm.parser_issue_code),
                            message=j_orm.parser_issue_message or "",
                            metadata=j_orm.parser_issue_metadata or {}
                        )

                    # Build LLMPromptText entity
                    llm_prompt_text = LLMPromptText(
                        id=j_orm.llm_prompt_text.id,
                        prompt_text=j_orm.llm_prompt_text.prompt_text,
                        content_hash=j_orm.llm_prompt_text.content_hash,
                    )

                    # Build LLMResponseText entity
                    llm_response_text = LLMResponseText(
                        id=j_orm.llm_response_text.id,
                        llm_response_text=j_orm.llm_response_text.llm_response_text,
                        content_hash=j_orm.llm_response_text.content_hash,
                    )

                    # Build LLMJudgement
                    llm_judgement = LLMJudgement(
                        id=j_orm.id,
                        dataset_sample=dataset_sample,
                        llm_prompt_text=llm_prompt_text,
                        llm_response_text=llm_response_text,
                        llm_invocation_metrics=invocation_metrics,
                        llm_score=llm_score,
                        parser_issue=parser_issue,
                    )

                    llm_judgements.append(llm_judgement)

                # Create InferRunOutput domain object
                # Note: Aggregate metrics (judgement_count, error_count, avg_latency_ms) use defaults
                # These are not stored in the database and not needed for aggregation
                infer_run_output = InferRunOutput(
                    id=infer_run_output_orm.id,
                    sample_fingerprint=infer_run_output_orm.sample_fingerprint or "",
                    llm_judgements=llm_judgements,
                    judgement_count=len(llm_judgements),
                    finished=infer_run_output_orm.finished,
                )

                infer_run_outputs.append(infer_run_output)

            return infer_run_outputs
