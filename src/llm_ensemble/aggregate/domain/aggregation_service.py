"""Domain service for aggregation pipeline.

This module contains pure business logic for orchestrating ensemble aggregation.
It depends only on port abstractions and handles its own logging.
"""

from __future__ import annotations
from uuid import UUID
from collections import defaultdict

from llm_ensemble.infer.schemas.judged_dataset import JudgedDataset
from llm_ensemble.infer.schemas.dataset_judgement import DatasetJudgement
from llm_ensemble.aggregate.schemas import AggregatedDataset, DatasetVote, AggregatedVote
from llm_ensemble.aggregate.schemas.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.schemas.aggregate_run_summary import AggregateRunSummary
from llm_ensemble.aggregate.ports import (
    AggregationStrategy,
    JudgementReader,
)
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder


class AggregationService:
    """Domain service for coordinating ensemble aggregation pipeline.

    Pure business logic that orchestrates:
    - Reading JudgedDatasets via JudgementReader port
    - Validating fingerprints match (same samples processed)
    - Grouping dataset_judgements by sequence_number position
    - Applying aggregation strategy to each group
    - Creating AggregatedDataset with DatasetVotes and AggregatedVotes
    - Tracking statistics (ties, no-votes)

    New Architecture:
        The service works with the normalized aggregate schema:
        - AggregatedDataset: Idempotent set identified by fingerprint
        - DatasetVote: Single position in the dataset (like DatasetSample in ingest)
        - AggregatedVote: Result of applying aggregation strategy
        - AggregationVote: Tracks which dataset_judgements were aggregated

    Idempotency:
        Multiple aggregate runs aggregating the same votes will produce
        the same AggregatedDataset (same fingerprint → same UUID).
    """

    def __init__(
        self,
        judgement_reader: JudgementReader,
        strategy: AggregationStrategy,
        aggregation_spec_id: UUID,
    ):
        """Initialize aggregation service with port dependencies.

        Args:
            judgement_reader: Port for reading JudgedDataset records
            strategy: Port for aggregation strategy (e.g., MajorityVoteAdapter)
            aggregation_spec_id: UUID of the aggregation spec being used
        """
        self.judgement_reader = judgement_reader
        self.strategy = strategy
        self.aggregation_spec_id = aggregation_spec_id
        self.logger = get_logger(component="aggregation_service")

    def _validate_judged_datasets(
        self, judged_datasets: list[JudgedDataset], run_names: list[str]
    ) -> None:
        """Validate that all JudgedDatasets are complete and compatible for aggregation.

        Checks:
        1. All JudgedDatasets have non-NULL fingerprints (run completed successfully)
        2. All fingerprints match (same samples were processed)

        Args:
            judged_datasets: List of JudgedDataset objects loaded by reader
            run_names: Corresponding run names (for error messages)

        Raises:
            ValueError: If validation fails
        """
        if not judged_datasets:
            raise ValueError("No JudgedDatasets found. Cannot aggregate empty list.")

        # Check for NULL fingerprints (incomplete runs)
        for dataset, run_name in zip(judged_datasets, run_names):
            if dataset.fingerprint is None:
                raise ValueError(
                    f"JudgedDataset for run '{run_name}' has NULL fingerprint. "
                    f"This indicates the run did not complete successfully."
                )

        # Check that all fingerprints match
        fingerprints = {dataset.fingerprint for dataset in judged_datasets}
        if len(fingerprints) > 1:
            raise ValueError(
                f"Cannot aggregate runs with different JudgedDataset fingerprints. "
                f"Found {len(fingerprints)} distinct fingerprints. "
                f"This means the runs processed different sets of samples."
            )

    def run_aggregation(
        self,
        run_names: list[str],
        run_info: AggregateRunInfo,
        run_dir,
    ) -> AggregateRunSummary:
        """Execute the aggregation pipeline.

        Pure business logic that:
        1. Reads JudgedDatasets via reader port
        2. Validates fingerprints match
        3. For each sequence_number position:
           - Collects all dataset_judgements at that position from all runs
           - Extracts all llm_judgements from those dataset_judgements
           - Applies aggregation strategy to get consensus
           - Creates AggregatedVote with result
           - Creates DatasetVote with the AggregatedVote
        4. Creates AggregatedDataset from all DatasetVotes
        5. Tracks statistics (ties, no-votes, etc.)

        Args:
            run_names: List of infer run identifiers to read judgements from
            run_info: Immutable runtime context (attached to summary)
            run_dir: Run directory for output

        Returns:
            AggregateRunSummary with statistics
        """
        # Initialize summary builder with run_info
        summary_builder = RunSummaryBuilder(run_info)
        summary_builder.set_start_time()

        # Read JudgedDatasets (one per run) via reader port
        judged_datasets : list[JudgedDataset] = self.judgement_reader.read(run_names)

        # Validate completion and fingerprint consistency
        self._validate_judged_datasets(judged_datasets, run_names)

        # Log validation
        fingerprints = {dataset.fingerprint for dataset in judged_datasets}
        self.logger.info(
            "validated_judged_datasets",
            num_datasets=len(judged_datasets),
            shared_fingerprint=list(fingerprints)[0][:16] + "..." if fingerprints else "N/A"
        )

        # Group dataset_judgements by sequence_number across all runs
        # Key: sequence_number, Value: list of dataset_judgements from different runs at that position
        grouped_by_position: dict[int, list[DatasetJudgement]] = defaultdict(list)

        for judged_dataset in judged_datasets:
            for dataset_judgement in judged_dataset.dataset_judgements:
                grouped_by_position[dataset_judgement.sequence_number].append(dataset_judgement)

        # Track statistics
        tie_count = 0
        no_valid_votes_count = 0
        dataset_votes = []

        # Process each sequence_number position
        for sequence_number in sorted(grouped_by_position.keys()):
            # All dataset_judgements at this position (one from each run)
            dataset_judgements_at_position = grouped_by_position[sequence_number]

            # Extract all LLM judgements from all dataset_judgements at this position
            all_llm_judgements = []
            for dataset_judgement in dataset_judgements_at_position:
                all_llm_judgements.extend(dataset_judgement.llm_judgements)

            # Apply aggregation strategy to get consensus
            final_label, final_confidence, final_reasoning = self.strategy.aggregate(all_llm_judgements)

            # Track statistics
            if final_label is None:
                no_valid_votes_count += 1
            elif final_reasoning and "tie" in final_reasoning.lower():
                tie_count += 1

            # Compute UUIDs (placeholder aggregated_dataset_id for now)
            placeholder_aggregated_dataset_id = UUID(int=0)
            dataset_vote_id = compute_dataset_vote_uuid(
                placeholder_aggregated_dataset_id,
                sequence_number
            )
            aggregated_vote_id = compute_aggregated_vote_uuid(
                dataset_vote_id,
                self.aggregation_spec_id
            )

            # Create AggregatedVote with full dataset_judgements
            aggregated_vote = AggregatedVote(
                id=aggregated_vote_id,
                dataset_vote_id=dataset_vote_id,
                aggregation_spec_id=self.aggregation_spec_id,
                dataset_judgements=dataset_judgements_at_position,
                final_label=final_label,
                final_confidence=final_confidence,
                final_reasoning=final_reasoning,
            )

            # Create DatasetVote
            dataset_vote = DatasetVote(
                id=dataset_vote_id,
                aggregated_dataset_id=placeholder_aggregated_dataset_id,
                sequence_number=sequence_number,
                aggregated_votes=[aggregated_vote],
            )

            dataset_votes.append(dataset_vote)

            # Log progress
            self.logger.info(
                "aggregated_position",
                sequence_number=sequence_number,
                final_label=final_label.label if final_label else "None",
                confidence=f"{final_confidence:.2f}" if final_confidence else "0.00",
                num_llm_judgements=len(all_llm_judgements),
                num_dataset_judgements=len(dataset_judgements_at_position),
            )

        # Create AggregatedDataset (computes fingerprint and UUID)
        aggregated_dataset = AggregatedDataset.create(dataset_votes)

        self.logger.info(
            "created_aggregated_dataset",
            dataset_id=str(aggregated_dataset.id),
            fingerprint=aggregated_dataset.fingerprint[:16] + "...",
            vote_count=aggregated_dataset.vote_count,
        )

        # TODO: Write aggregated_dataset via writer port

        # Build and finalize summary
        total_llm_judgements = sum(
            len(dj.llm_judgements)
            for judged_dataset in judged_datasets
            for dj in judged_dataset.dataset_judgements
        )
        summary_builder.add("input_judgement_count", total_llm_judgements)
        summary_builder.add("unique_pair_count", len(grouped_by_position))
        summary_builder.add("output_aggregated_count", len(dataset_votes))
        summary_builder.add("tie_count", tie_count)
        summary_builder.add("no_valid_votes_count", no_valid_votes_count)

        # Optional: add warnings summary
        warnings_summary = {}
        if tie_count > 0:
            warnings_summary["tie"] = tie_count
        if no_valid_votes_count > 0:
            warnings_summary["no_valid_votes"] = no_valid_votes_count
        if warnings_summary:
            summary_builder.add("warnings_summary", warnings_summary)

        return summary_builder.finalize(AggregateRunSummary)
