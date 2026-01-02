"""SQL writer adapter for persisting aggregate runs to database.

Uses pure SQLAlchemy ORM models with random UUIDs.
Duplicate detection via pre-querying existing entities (no exception handling).
Handles its own logging and returns write summary as metadata.

No mapper layer - direct ORM conversion for bidirectional symmetry.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import tuple_

from llm_ensemble.aggregate.domain.entities.aggregate_run import AggregateRun
from llm_ensemble.aggregate.domain.entities.aggregate_run_config import AggregateRunConfig
from llm_ensemble.aggregate.domain.entities.aggregated_dataset import AggregatedDataset
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.domain.entities.write_summary import WriteSummary
from llm_ensemble.aggregate.adapters.driven.io.orms import (
    AggregationStrategyORM,
    AggregateRunConfigORM,
    AggregateRunORM,
    AggregatedDatasetORM,
    AggregatedVoteORM,
    AggregationVoteORM,
    AggregatedDatasetVoteORM,
)
from llm_ensemble.aggregate.application.ports.driven.for_output import ForOutput
from llm_ensemble.libs.logging.structlog_logger import get_logger
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context
from llm_ensemble.libs.logging.log_events import AggregateWriteEvent


class DBWriter(ForOutput):
    """SQL writer adapter for aggregate runs - handles ORM mapping.

    Writes aggregate runs to SQL database using pure SQLAlchemy ORM.
    Contains the mapping layer that extracts dataset/run context from aggregate_run
    and handles ORM relationships.

    Features:
    - Duplicate detection via pre-querying existing entities by natural keys
    - Bulk insert operations for maximum performance
    - UUID mapping to ensure correct foreign key references across runs
    - Natural key deduplication (name for Strategy, fingerprint for Dataset)
    - Uses session_context() for transaction management
    - Logs write operations directly

    Database URL is read from DATABASE_URL environment variable (required).
    Example: postgresql://user:password@localhost:5432/llm_ensemble
    """

    def __init__(self, io_name: str, database_url: str | None = None):
        """Initialize SQL writer with IO format name.

        Args:
            io_name: Name of the IO format (e.g., 'db_to_db')
            database_url: Optional database URL (defaults to DATABASE_URL env var)
        """
        self._io_name = io_name
        self.database_url = database_url
        self.engine = get_engine(database_url)
        self.logger = get_logger(component=__name__)

    @property
    def io_name(self) -> str:
        """Get I/O adapter name."""
        return self._io_name

    def write(self, aggregate_run: AggregateRun) -> WriteSummary:
        """Write aggregate run results to SQL database with direct logging.

        Duplicate detection via pre-querying existing entities by natural keys.
        Uses bulk insert operations for aggregated votes and junctions.
        Maintains UUID mappings to ensure correct foreign key references.
        Tracks created vs skipped entities in WriteSummary.
        Logs each entity type write and summary.

        Args:
            aggregate_run: Complete AggregateRun aggregate containing config, dataset, and metadata

        Returns:
            WriteSummary as pure data (metadata for run summary)

        Raises:
            IOError: If database write fails
        """
        # Extract dataset from aggregate
        aggregated_dataset = aggregate_run.aggregated_dataset
        aggregated_votes = aggregated_dataset.aggregated_votes

        if not aggregated_votes:
            return WriteSummary()

        # Extract aggregation strategy from run config
        aggregation_strategy = aggregate_run.aggregate_run_config.aggregation_strategy

        # Note: Tables must be created via `make db-init` before first write
        # Create summary builder
        summary = WriteSummary()

        # Write to database in transaction
        try:
            with session_context(self.engine) as session:
                # Save entities in strict dependency order to satisfy foreign key constraints
                # Order matters! Each step depends on previous steps being persisted.
                #
                # Dependency graph:
                #   1. AggregationStrategy (no dependencies - global entity)
                #   2. AggregateRunConfig (depends on AggregationStrategy)
                #   3. AggregatedDataset entity (no dependencies)
                #   4. AggregatedVote (no FK dependencies - pure data)
                #   5. AggregationVote junction (depends on AggregatedVote + LLMJudgement)
                #   6. AggregatedDatasetVote junction (depends on AggregatedDataset + AggregatedVote)
                #   7. AggregateRun (depends on AggregateRunConfig + AggregatedDataset)

                # 1. AggregationStrategy (no dependencies - global entity)
                # Returns actual DB UUID for use in foreign keys
                created, skipped, strategy_uuid = self._save_aggregation_strategy(session, aggregation_strategy)
                summary.add_aggregation_strategies(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(AggregateWriteEvent.WRITE_STRATEGY, created=created, skipped=skipped)

                # 2. AggregateRunConfig (depends on AggregationStrategy)
                created, skipped, config_uuid = self._save_aggregate_run_config(
                    session, aggregate_run.aggregate_run_config, strategy_uuid
                )
                summary.add_configs(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(AggregateWriteEvent.WRITE_RUN_CONFIG, created=created, skipped=skipped)

                # 3. AggregatedDataset entity (no dependencies)
                created, skipped, dataset_uuid = self._save_aggregated_dataset_entity(session, aggregated_dataset)
                summary.add_aggregated_datasets(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(AggregateWriteEvent.WRITE_DATASET, created=created, skipped=skipped)

                # Flush to ensure entities are persisted before creating dependent records
                session.flush()

                # 4. AggregatedVote (no FK dependencies - pure data)
                created, skipped, vote_uuid_map = self._save_aggregated_votes(
                    session, aggregated_votes
                )
                summary.add_aggregated_votes(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(AggregateWriteEvent.WRITE_VOTES, created=created, skipped=skipped)

                # 5. AggregationVote junction (depends on AggregatedVote + LLMJudgement)
                # Use vote_uuid_map to get actual DB UUIDs
                created, skipped = self._save_aggregation_votes(session, aggregated_votes, vote_uuid_map)
                summary.add_aggregation_votes(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(AggregateWriteEvent.WRITE_AGGREGATION_VOTES, created=created, skipped=skipped)

                # 6. AggregatedDatasetVote junction (depends on AggregatedDataset + AggregatedVote)
                # Use vote_uuid_map to get actual DB UUIDs
                created, skipped = self._save_aggregated_dataset_votes(
                    session, aggregated_votes, dataset_uuid, vote_uuid_map
                )
                summary.add_aggregated_dataset_votes(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(AggregateWriteEvent.WRITE_DATASET_VOTES, created=created, skipped=skipped)

                # 7. AggregateRun (depends on AggregateRunConfig + AggregatedDataset)
                created, skipped = self._save_aggregate_run(
                    session, aggregate_run, config_uuid, dataset_uuid
                )
                summary.add_aggregate_runs(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(AggregateWriteEvent.WRITE_RUNS, created=created, skipped=skipped)

            # Log totals
            if summary.total_created > 0 or summary.total_skipped > 0:
                self.logger.info(
                    AggregateWriteEvent.WRITE_COMPLETE,
                    total_created=summary.total_created,
                    total_skipped=summary.total_skipped,
                )

            return summary

        except Exception as e:
            raise IOError(f"Failed to write aggregate run to database: {e}") from e

    def _save_aggregation_strategy(
        self, session: Session, aggregation_strategy: AggregationStrategy
    ) -> Tuple[int, int, str]:
        """Save aggregation strategy entity to database using pre-query pattern.

        Checks if name already exists before inserting.
        Returns actual database UUID for foreign key references.

        Args:
            session: SQLAlchemy session
            aggregation_strategy: AggregationStrategy domain entity

        Returns:
            Tuple of (created_count, skipped_count, actual_db_uuid_as_string)
        """
        # Check if this strategy name already exists
        existing = (
            session.query(AggregationStrategyORM)
            .filter_by(name=aggregation_strategy.name)
            .first()
        )

        if existing:
            return (0, 1, str(existing.id))

        # Insert new strategy
        strategy_orm = AggregationStrategyORM(
            id=aggregation_strategy.id,
            name=aggregation_strategy.name,
        )
        session.add(strategy_orm)
        session.flush()
        return (1, 0, str(strategy_orm.id))

    def _save_aggregate_run_config(
        self, session: Session, run_config: AggregateRunConfig, strategy_uuid: str
    ) -> Tuple[int, int, str]:
        """Save aggregate run config entity to database using pre-query pattern.

        Checks if natural key already exists.
        Returns UUID of existing or newly created config.

        Args:
            session: SQLAlchemy session
            run_config: AggregateRunConfig domain object
            strategy_uuid: Actual database UUID of the aggregation strategy (as string)

        Returns:
            Tuple of (created_count, skipped_count, config_uuid)
        """
        # Check if this config already exists (by natural key)
        existing = (
            session.query(AggregateRunConfigORM)
            .filter_by(
                aggregation_strategy_id=strategy_uuid,
                io_config_name=run_config.io_config_name,
                input_run_names_hash=run_config.input_run_names_hash,
            )
            .first()
        )

        if existing:
            return (0, 1, str(existing.id))

        # Insert new config
        config_orm = AggregateRunConfigORM(
            id=run_config.id,
            aggregation_strategy_id=strategy_uuid,
            io_config_name=run_config.io_config_name,
            input_run_names=run_config.input_run_names,
            input_run_names_hash=run_config.input_run_names_hash,
        )
        session.add(config_orm)
        session.flush()
        return (1, 0, str(config_orm.id))

    def _save_aggregated_dataset_entity(
        self, session: Session, aggregated_dataset: AggregatedDataset
    ) -> Tuple[int, int, str]:
        """Save AggregatedDataset entity.

        Checks if fingerprint already exists before inserting.
        Returns UUID of existing or newly created dataset.

        Args:
            session: SQLAlchemy session
            aggregated_dataset: AggregatedDataset domain object

        Returns:
            Tuple of (created_count, skipped_count, dataset_uuid)
        """
        # Check if this dataset fingerprint already exists
        existing = (
            session.query(AggregatedDatasetORM)
            .filter_by(fingerprint=aggregated_dataset.fingerprint)
            .first()
        )

        if existing:
            return (0, 1, str(existing.id))

        # Insert new dataset
        dataset_orm = AggregatedDatasetORM(
            id=aggregated_dataset.id,
            fingerprint=aggregated_dataset.fingerprint,
        )
        session.add(dataset_orm)
        session.flush()
        return (1, 0, str(dataset_orm.id))

    def _save_aggregated_votes(
        self,
        session: Session,
        aggregated_votes: List[AggregatedVote],
    ) -> Tuple[int, int, Dict[UUID, str]]:
        """Save aggregated vote entities to database using bulk insert with pre-filtering.

        Queries existing votes by natural key (judgement_fingerprint, final_label, final_confidence, final_reasoning),
        filters to new ones, then bulk inserts. Returns mapping of vote domain UUID -> actual UUID in DB.

        Args:
            session: SQLAlchemy session
            aggregated_votes: List of AggregatedVote domain objects

        Returns:
            Tuple of (created_count, skipped_count, vote_uuid_mapping)
        """
        if not aggregated_votes:
            return (0, 0, {})

        # Import here to avoid circular dependency
        from hashlib import sha256
        import json

        # Convert votes to ORM objects and compute judgement fingerprints
        vote_orms: List[AggregatedVoteORM] = []
        for vote in aggregated_votes:
            # Compute judgement fingerprint from sorted judgement IDs
            sorted_judgement_ids = sorted([str(j.id) for j in vote.llm_judgements])
            canonical = json.dumps(sorted_judgement_ids, sort_keys=True)
            judgement_fingerprint = sha256(canonical.encode()).hexdigest()

            orm = AggregatedVoteORM(
                id=vote.id,
                judgement_fingerprint=judgement_fingerprint,
                final_label=vote.final_label,
                final_confidence=vote.final_confidence,
                final_reasoning=vote.final_reasoning,
            )
            vote_orms.append(orm)

        # Build list of natural key tuples to check
        vote_keys = [
            (orm.judgement_fingerprint, orm.final_label, orm.final_confidence, orm.final_reasoning)
            for orm in vote_orms
        ]

        # Query existing votes and get their UUIDs
        existing_votes = {
            (fp, label, conf, reasoning): str(vote_id)
            for (fp, label, conf, reasoning, vote_id) in
            session.query(
                AggregatedVoteORM.judgement_fingerprint,
                AggregatedVoteORM.final_label,
                AggregatedVoteORM.final_confidence,
                AggregatedVoteORM.final_reasoning,
                AggregatedVoteORM.id
            )
            .filter(
                tuple_(
                    AggregatedVoteORM.judgement_fingerprint,
                    AggregatedVoteORM.final_label,
                    AggregatedVoteORM.final_confidence,
                    AggregatedVoteORM.final_reasoning
                )
                .in_(vote_keys)
            )
        }

        # Filter to only new votes
        new_vote_orms = [
            orm for orm in vote_orms
            if (orm.judgement_fingerprint, orm.final_label, orm.final_confidence, orm.final_reasoning) not in existing_votes
        ]

        # Bulk insert new votes
        if new_vote_orms:
            session.add_all(new_vote_orms)
            session.flush()

        # Build complete mapping: domain vote UUID -> actual UUID in DB (existing + new)
        vote_uuid_map = {}
        for vote, orm in zip(aggregated_votes, vote_orms):
            vote_key = (orm.judgement_fingerprint, orm.final_label, orm.final_confidence, orm.final_reasoning)
            if vote_key in existing_votes:
                # Vote exists - use actual database UUID from existing record
                vote_uuid_map[vote.id] = existing_votes[vote_key]
            else:
                # New vote - use ORM's UUID (just inserted)
                vote_uuid_map[vote.id] = str(orm.id)

        created = len(new_vote_orms)
        skipped = len(vote_orms) - created

        return (created, skipped, vote_uuid_map)

    def _save_aggregation_votes(
        self,
        session: Session,
        aggregated_votes: List[AggregatedVote],
        vote_uuid_map: Dict[UUID, str],
    ) -> Tuple[int, int]:
        """Save AggregationVote junction records using bulk insert with pre-filtering.

        Links AggregatedVote to LLMJudgements.

        Args:
            session: SQLAlchemy session
            aggregated_votes: List of AggregatedVote domain objects
            vote_uuid_map: Maps domain vote UUID to actual DB UUID

        Returns:
            Tuple of (created_count, skipped_count)
        """
        # Create all junction records
        junction_orms: List[AggregationVoteORM] = []
        for vote in aggregated_votes:
            # Use actual DB UUID from map, not domain UUID
            actual_vote_id = vote_uuid_map[vote.id]
            for llm_judgement in vote.llm_judgements:
                junction = AggregationVoteORM(
                    aggregated_vote_id=actual_vote_id,
                    llm_judgement_id=llm_judgement.id,
                )
                junction_orms.append(junction)

        if not junction_orms:
            return (0, 0)

        # Build list of (vote_id, judgement_id) tuples to check
        junction_keys = [(j.aggregated_vote_id, j.llm_judgement_id) for j in junction_orms]

        # Query existing junctions
        existing_junctions = {
            (str(vote_id), str(judgement_id))
            for (vote_id, judgement_id) in
            session.query(
                AggregationVoteORM.aggregated_vote_id,
                AggregationVoteORM.llm_judgement_id
            )
            .filter(
                tuple_(AggregationVoteORM.aggregated_vote_id, AggregationVoteORM.llm_judgement_id)
                .in_(junction_keys)
            )
        }

        # Filter to only new junctions
        new_junction_orms = [
            j for j in junction_orms
            if (str(j.aggregated_vote_id), str(j.llm_judgement_id)) not in existing_junctions
        ]

        # Bulk insert new junctions
        if new_junction_orms:
            session.add_all(new_junction_orms)
            session.flush()

        created = len(new_junction_orms)
        skipped = len(junction_orms) - created

        return (created, skipped)

    def _save_aggregated_dataset_votes(
        self,
        session: Session,
        aggregated_votes: List[AggregatedVote],
        dataset_uuid: str,
        vote_uuid_map: Dict[UUID, str],
    ) -> Tuple[int, int]:
        """Save AggregatedDatasetVote junction records using bulk insert with pre-filtering.

        Links AggregatedDataset to AggregatedVotes.

        Args:
            session: SQLAlchemy session
            aggregated_votes: List of AggregatedVote domain objects
            dataset_uuid: UUID of the AggregatedDataset in database
            vote_uuid_map: Maps domain vote UUID to actual DB UUID

        Returns:
            Tuple of (created_count, skipped_count)
        """
        # Create all junction records
        junction_orms: List[AggregatedDatasetVoteORM] = []
        for vote in aggregated_votes:
            # Use actual DB UUID from map, not domain UUID
            actual_vote_id = vote_uuid_map[vote.id]
            junction = AggregatedDatasetVoteORM(
                aggregated_dataset_id=dataset_uuid,
                aggregated_vote_id=actual_vote_id,
            )
            junction_orms.append(junction)

        if not junction_orms:
            return (0, 0)

        # Build list of (dataset_id, vote_id) tuples to check
        junction_keys = [(j.aggregated_dataset_id, j.aggregated_vote_id) for j in junction_orms]

        # Query existing junctions
        existing_junctions = {
            (str(dataset_id), str(vote_id))
            for (dataset_id, vote_id) in
            session.query(
                AggregatedDatasetVoteORM.aggregated_dataset_id,
                AggregatedDatasetVoteORM.aggregated_vote_id
            )
            .filter(
                tuple_(AggregatedDatasetVoteORM.aggregated_dataset_id, AggregatedDatasetVoteORM.aggregated_vote_id)
                .in_(junction_keys)
            )
        }

        # Filter to only new junctions
        new_junction_orms = [
            j for j in junction_orms
            if (str(j.aggregated_dataset_id), str(j.aggregated_vote_id)) not in existing_junctions
        ]

        # Bulk insert new junctions
        if new_junction_orms:
            session.add_all(new_junction_orms)
            session.flush()

        created = len(new_junction_orms)
        skipped = len(junction_orms) - created

        return (created, skipped)

    def _save_aggregate_run(
        self,
        session: Session,
        aggregate_run: AggregateRun,
        config_uuid: str,
        dataset_uuid: str,
    ) -> Tuple[int, int]:
        """Save aggregate run entity to database using pre-query pattern.

        Checks if run_name already exists before inserting.
        Uses provided config and dataset UUIDs for correct foreign key references.

        Args:
            session: SQLAlchemy session
            aggregate_run: AggregateRun aggregate (contains config and dataset)
            config_uuid: UUID of the AggregateRunConfig in database
            dataset_uuid: UUID of the AggregatedDataset in database

        Returns:
            Tuple of (created_count, skipped_count)
        """
        # Check if this run_name already exists
        existing = session.query(AggregateRunORM).filter_by(run_name=aggregate_run.run_name).first()

        if existing:
            return (0, 1)

        # Insert new run with correct foreign key UUIDs
        aggregate_run_orm = AggregateRunORM(
            id=aggregate_run.id,
            run_name=aggregate_run.run_name,
            run_type=aggregate_run.run_type,
            aggregate_run_config_id=config_uuid,
            aggregated_dataset_id=dataset_uuid,
            start_time=aggregate_run.start_time,
            end_time=aggregate_run.end_time,
            git_sha=aggregate_run.git_info.git_sha,
            git_branch=aggregate_run.git_info.git_branch,
            git_is_dirty="true" if not aggregate_run.git_info.git_clean else "false",
            notes=aggregate_run.notes,
        )
        session.add(aggregate_run_orm)
        session.flush()
        return (1, 0)
