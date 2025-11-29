"""Database adapter for writing aggregated datasets.

Writes AggregatedDataset records to PostgreSQL database using SQLAlchemy ORM.
Decomposes denormalized domain objects into normalized relational entities.

Uses data mapper pattern: domain service works with AggregatedDataset objects,
SQL writer maps them to ORM entities. Mapper logic lives in mappers_domain_to_orm.py.

Architecture:
- Run metadata initialized once in open()
- Per-vote entities created in write_one() using mappers
- Immediate commits for fault tolerance
- AggregatedDataset finalized in close()
"""

from __future__ import annotations
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from llm_ensemble.aggregate.schemas.aggregated_dataset import AggregatedDataset
from llm_ensemble.aggregate.schemas.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.schemas.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.schemas.write_summary import WriteSummary
from llm_ensemble.aggregate.schemas.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.ports import AggregatedJudgementWriter
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.db import (
    get_engine,
    get_session,
    compute_aggregate_run_uuid,
)
from llm_ensemble.aggregate.schemas.orms_normalized import (
    AggregationStrategyORM,
    AggregateRunORM,
    AggregatedDatasetORM,
    AggregatedVoteORM,
    AggregationVoteORM,
    AggregatedDatasetVoteORM,
)
from llm_ensemble.aggregate.adapters.io.mappers_domain_to_orm import (
    aggregation_strategy_to_orm,
    aggregate_run_info_to_orm,
    aggregated_dataset_to_orm,
    aggregated_vote_to_orm,
    create_aggregation_vote_orm,
    create_aggregated_dataset_vote_orm,
)


class DbAggregatedDatasetWriter(AggregatedJudgementWriter):
    """Write AggregatedDataset records to SQL database.

    Normalized schema: decomposes aggregated datasets into AggregationSpec,
    AggregateRun, AggregatedDataset, AggregatedVote, AggregationVote,
    and AggregatedDatasetVote entities.

    Deduplication via deterministic UUIDs + unique constraints.
    """

    def __init__(self):
        self.logger = get_logger(component="db_aggregated_dataset_writer")

    def write(
        self,
        run_dir: Path,
        run_info: AggregateRunInfo,
        aggregated_dataset: AggregatedDataset,
    ) -> WriteSummary:
        """Write entire aggregated dataset to database in one batch.

        Args:
            run_dir: Run directory (unused for database writes)
            run_info: Aggregate run context
            aggregated_dataset: The aggregated dataset to write

        Returns:
            WriteSummary tracking what entities were created/skipped
        """
        write_summary = WriteSummary()

        engine = get_engine()
        session = get_session(engine)

        try:
            # Upsert AggregationStrategy (extract from first aggregated_vote)
            if aggregated_dataset.aggregated_votes:
                aggregation_strategy = aggregated_dataset.aggregated_votes[0].aggregation_strategy
                self._upsert_aggregation_strategy(session, aggregation_strategy, write_summary)

            # Initialize run metadata (aggregate run)
            aggregate_run_id = self._initialize_run_metadata(
                session, run_info, write_summary
            )

            # Upsert AggregatedDataset
            aggregated_dataset_id = self._upsert_aggregated_dataset(
                session, aggregated_dataset, write_summary
            )

            self.logger.info(
                "writing_aggregated_dataset",
                aggregated_dataset_id=str(aggregated_dataset_id),
                fingerprint=aggregated_dataset.fingerprint[:16] + "...",
                vote_count=len(aggregated_dataset.aggregated_votes),
            )

            # Write all aggregated votes
            for aggregated_vote in aggregated_dataset.aggregated_votes:
                self._write_aggregated_vote(
                    session,
                    aggregated_dataset_id,
                    aggregated_vote,
                    write_summary
                )

            # Link AggregateRun to AggregatedDataset (finalize relationship)
            aggregate_run = session.get(AggregateRunORM, aggregate_run_id)
            if aggregate_run:
                aggregate_run.aggregated_dataset_id = aggregated_dataset_id
                session.commit()

            # Log totals
            self.logger.info(
                "write_complete",
                total_created=write_summary.total_created,
                total_skipped=write_summary.total_skipped,
            )

            return write_summary

        finally:
            session.close()

    def _upsert_aggregation_strategy(
        self,
        session: Session,
        aggregation_strategy: "AggregationStrategy",
        write_summary: WriteSummary,
    ) -> None:
        """Upsert AggregationStrategy entity.

        Args:
            session: Database session
            aggregation_strategy: AggregationStrategy domain entity
            write_summary: Summary tracker
        """
        strategy_id = self._upsert_entity(
            session,
            AggregationStrategyORM,
            aggregation_strategy.id,
            lambda: aggregation_strategy_to_orm(aggregation_strategy),
            "aggregation_strategies",
            write_summary
        )

    def _initialize_run_metadata(
        self,
        session: Session,
        run_info: AggregateRunInfo,
        write_summary: WriteSummary,
    ) -> uuid.UUID:
        """Initialize run metadata.

        Returns:
            aggregate_run_id
        """
        # Create AggregateRun
        aggregate_run_id = compute_aggregate_run_uuid(run_info.run_name)
        config_names = {
            "aggregation_strategy_adapter": run_info.aggregation_strategy_adapter_spec.name,
            "io_config": run_info.io_config_name,
        }
        aggregate_run_orm = aggregate_run_info_to_orm(
            run_info,
            aggregate_run_id,
            config_names
        )
        session.add(aggregate_run_orm)
        write_summary.add_aggregate_runs(created=1)
        session.commit()

        return aggregate_run_id

    def _upsert_entity(
        self,
        session: Session,
        orm_class,
        entity_id: uuid.UUID,
        create_fn,
        entity_name: str,
        write_summary: WriteSummary,
    ) -> uuid.UUID:
        """Generic upsert helper."""
        existing = session.get(orm_class, entity_id)
        if existing:
            # Track skip
            attr_name = f"add_{entity_name}"
            if hasattr(write_summary, attr_name):
                getattr(write_summary, attr_name)(created=0, skipped=1)
            return entity_id

        entity_orm = create_fn()
        session.add(entity_orm)

        # Track create
        attr_name = f"add_{entity_name}"
        if hasattr(write_summary, attr_name):
            getattr(write_summary, attr_name)(created=1, skipped=0)

        return entity_id

    def _upsert_aggregated_dataset(
        self,
        session: Session,
        aggregated_dataset: AggregatedDataset,
        write_summary: WriteSummary,
    ) -> uuid.UUID:
        """Upsert AggregatedDataset from domain object."""
        aggregated_dataset_orm = aggregated_dataset_to_orm(aggregated_dataset)
        existing = session.get(AggregatedDatasetORM, aggregated_dataset.id)

        if existing:
            write_summary.add_aggregated_datasets(created=0, skipped=1)
            return aggregated_dataset.id

        session.add(aggregated_dataset_orm)
        write_summary.add_aggregated_datasets(created=1, skipped=0)
        session.commit()
        return aggregated_dataset.id

    def _write_aggregated_vote(
        self,
        session: Session,
        aggregated_dataset_id: uuid.UUID,
        aggregated_vote: AggregatedVote,
        write_summary: WriteSummary,
    ) -> None:
        """Write a single aggregated vote with all junction records."""
        # Upsert AggregatedVote
        aggregated_vote_orm = aggregated_vote_to_orm(aggregated_vote)
        existing = session.get(AggregatedVoteORM, aggregated_vote.id)

        if not existing:
            session.add(aggregated_vote_orm)
            write_summary.add_aggregated_votes(created=1, skipped=0)
        else:
            write_summary.add_aggregated_votes(created=0, skipped=1)

        # Create AggregationVote junction records (link to llm_judgements)
        for llm_judgement in aggregated_vote.llm_judgements:
            aggregation_vote_orm = create_aggregation_vote_orm(
                aggregated_vote.id,
                llm_judgement.id
            )
            # Upsert (handle duplicates)
            existing_junction = session.query(AggregationVoteORM).filter_by(
                aggregated_vote_id=aggregated_vote.id,
                llm_judgement_id=llm_judgement.id
            ).one_or_none()

            if not existing_junction:
                session.add(aggregation_vote_orm)
                write_summary.add_aggregation_votes(created=1)
            else:
                write_summary.add_aggregation_votes(skipped=1)

        # Create AggregatedDatasetVote junction record (link dataset to vote)
        aggregated_dataset_vote_orm = create_aggregated_dataset_vote_orm(
            aggregated_dataset_id,
            aggregated_vote.id
        )
        # Upsert (handle duplicates)
        existing_dataset_vote = session.query(AggregatedDatasetVoteORM).filter_by(
            aggregated_dataset_id=aggregated_dataset_id,
            aggregated_vote_id=aggregated_vote.id
        ).one_or_none()

        if not existing_dataset_vote:
            session.add(aggregated_dataset_vote_orm)
            write_summary.add_aggregated_dataset_votes(created=1)

        session.commit()
