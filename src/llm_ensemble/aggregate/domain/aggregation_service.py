"""Domain service for aggregation pipeline.

This module contains pure business logic for orchestrating ensemble aggregation.
It depends only on port abstractions and handles its own logging.
"""

from __future__ import annotations
from uuid import UUID
from collections import defaultdict

from llm_ensemble.infer.schemas.entities.judged_dataset import JudgedDataset
from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas import AggregatedDataset, AggregatedVote
from llm_ensemble.aggregate.schemas.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.schemas.aggregate_run_summary import AggregateRunSummary
from llm_ensemble.aggregate.ports import (
    AggregatedJudgementWriter,
    AggregationStrategyPort,
    JudgementReader,
)
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder


class AggregationService:
    """Domain service for coordinating ensemble aggregation pipeline.

    Pure business logic that orchestrates:
    - Reading JudgedDatasets via JudgementReader port
    - Validating sample_fingerprints match (same samples processed)
    - Grouping LLM judgements by dataset_sample_id across runs
    - Applying aggregation strategy to each group
    - Creating AggregatedDataset with DatasetVotes and AggregatedVotes
    - Writing AggregatedDataset via writer port
    - Tracking statistics (ties, no-votes)

    New Architecture:
        The service works with the normalized aggregate schema:
        - AggregatedDataset: Idempotent set identified by fingerprint
        - DatasetVote: Single position in the dataset (like DatasetSample in ingest)
        - AggregatedVote: Result of applying aggregation strategy
        - AggregationVote: Tracks which llm_judgements were aggregated

    Idempotency:
        Multiple aggregate runs aggregating the same votes will produce
        the same AggregatedDataset (same fingerprint → same UUID).
    """

    def __init__(
        self,
        judgement_reader: JudgementReader,
        aggregated_judgement_writer: AggregatedJudgementWriter,
        aggregation_strategy_adapter: AggregationStrategyPort,
    ):
        """Initialize aggregation service with port dependencies.

        Args:
            judgement_reader: Port for reading JudgedDataset records
            aggregated_judgement_writer: Port for writing AggregatedDataset records
            strategy: Port for aggregation strategy (e.g., MajorityVoteAdapter)
        """
        self.judgement_reader = judgement_reader
        self.aggregated_judgement_writer = aggregated_judgement_writer
        self.strategy = aggregation_strategy_adapter
        self.logger = get_logger(component="aggregation_service")

    def _validate_judged_datasets(
        self, judged_datasets: list[JudgedDataset], run_names: list[str]
    ) -> None:
        """Validate that all JudgedDatasets are complete and compatible for aggregation.

        Checks:
        1. All JudgedDatasets have non-NULL sample_fingerprints (run completed successfully)
        2. All sample_fingerprints match (same samples were processed)

        Args:
            judged_datasets: List of JudgedDataset objects loaded by reader
            run_names: Corresponding run names (for error messages)

        Raises:
            ValueError: If validation fails
        """
        if not judged_datasets:
            raise ValueError("No JudgedDatasets found. Cannot aggregate empty list.")

        # Check for NULL sample_fingerprints (incomplete runs)
        for dataset, run_name in zip(judged_datasets, run_names):
            if dataset.sample_fingerprint is None:
                raise ValueError(
                    f"JudgedDataset for run '{run_name}' has NULL sample_fingerprint. "
                    f"This indicates the run did not complete successfully."
                )

        # Check that all sample_fingerprints match
        sample_fingerprints = {dataset.sample_fingerprint for dataset in judged_datasets}
        if len(sample_fingerprints) > 1:
            raise ValueError(
                f"Cannot aggregate runs with different JudgedDataset sample_fingerprints. "
                f"Found {len(sample_fingerprints)} distinct sample_fingerprints. "
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
        2. Validates sample_fingerprints match
        3. For each dataset_sample_id:
           - Collects all llm_judgements for that sample from all runs
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
        # Initialize summary builder
        summary_builder = RunSummaryBuilder()
        summary_builder.set_start_time()
        summary_builder.add("run_info", run_info)

        # Read JudgedDatasets (one per run) via reader port
        judged_datasets : list[JudgedDataset] = self.judgement_reader.read(run_names)

        # Validate completion and sample_fingerprint consistency
        self._validate_judged_datasets(judged_datasets, run_names)

        # Log validation
        sample_fingerprints = {dataset.sample_fingerprint for dataset in judged_datasets}
        self.logger.info(
            "validated_judged_datasets",
            num_datasets=len(judged_datasets),
            shared_sample_fingerprint=list(sample_fingerprints)[0][:16] + "..." if sample_fingerprints else "N/A"
        )

        # Group llm_judgements by dataset_sample_id across all runs
        # Key: dataset_sample_id, Value: list of llm_judgements from different runs for that sample
        grouped_by_sample: dict[UUID, list[LLMJudgement]] = defaultdict(list)

        for judged_dataset in judged_datasets:
            for llm_judgement in judged_dataset.llm_judgements:
                # Get dataset_sample_id via llm_prompt → dataset_sample
                dataset_sample_id = llm_judgement.llm_prompt.dataset_sample.id
                grouped_by_sample[dataset_sample_id].append(llm_judgement)

        # Track statistics
        tie_count = 0
        no_valid_votes_count = 0
        aggregated_votes = []

        # Process each dataset_sample_id
        for dataset_sample_id in sorted(grouped_by_sample.keys()):
            # All llm_judgements for this sample (one from each run/model config)
            llm_judgements_for_sample = grouped_by_sample[dataset_sample_id]

            # Apply aggregation strategy to get AggregatedVote
            aggregated_vote : AggregatedVote = self.strategy.aggregate(llm_judgements_for_sample)

            # Track statistics
            if aggregated_vote.final_label is None:
                no_valid_votes_count += 1
            elif aggregated_vote.final_reasoning and "tie" in aggregated_vote.final_reasoning.lower():
                tie_count += 1

            aggregated_votes.append(aggregated_vote)

            # Log progress
            self.logger.info(
                "aggregated_sample",
                dataset_sample_id=str(dataset_sample_id)[:8] + "...",
                final_label=aggregated_vote.final_label.label if aggregated_vote.final_label else "None",
                confidence=f"{aggregated_vote.final_confidence:.2f}" if aggregated_vote.final_confidence else "0.00",
                num_llm_judgements=len(llm_judgements_for_sample),
            )

        # Create AggregatedDataset (computes fingerprint and UUID from votes)
        aggregated_dataset = AggregatedDataset.create(aggregated_votes)

        self.logger.info(
            "created_aggregated_dataset",
            dataset_id=str(aggregated_dataset.id),
            fingerprint=aggregated_dataset.fingerprint[:16] + "...",
            vote_count=aggregated_dataset.vote_count,
        )

        # Write aggregated_dataset via writer port
        write_summary = self.aggregated_judgement_writer.write(
            run_dir=run_dir,
            run_info=run_info,
            aggregated_dataset=aggregated_dataset,
        )

        self.logger.info(
            "write_complete",
            entities_created=write_summary.total_created,
            entities_skipped=write_summary.total_skipped,
        )

        # Build and finalize summary
        total_llm_judgements = sum(
            len(judged_dataset.llm_judgements)
            for judged_dataset in judged_datasets
        )
        summary_builder.add("input_judgement_count", total_llm_judgements)
        summary_builder.add("unique_pair_count", len(grouped_by_sample))
        summary_builder.add("output_aggregated_count", len(aggregated_votes))
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
