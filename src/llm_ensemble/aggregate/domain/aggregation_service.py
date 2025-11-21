"""Domain service for aggregation pipeline.

This module contains pure business logic for orchestrating ensemble aggregation.
It depends only on port abstractions and handles its own logging.
"""

from __future__ import annotations
from collections import defaultdict

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.judged_dataset import JudgedDataset
from llm_ensemble.aggregate.schemas import AggregatedJudgement
from llm_ensemble.aggregate.schemas.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.schemas.aggregate_run_summary import AggregateRunSummary
from llm_ensemble.aggregate.ports import (
    AggregationStrategy,
    JudgementReader,
    AggregatedJudgementWriter,
)
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder


class AggregationService:
    """Domain service for coordinating ensemble aggregation pipeline.
    
    Pure business logic that orchestrates:
    - Reading judgements via JudgementReader port
    - Grouping judgements by natural composite key (dataset, query external_id, doc external_id)
    - Applying aggregation strategy to each group
    - Writing aggregated judgements via AggregatedJudgementWriter port
    - Tracking statistics (ties, no-votes)
    
    Depends only on port abstractions and handles its own logging, enabling complete
    independence from infrastructure concerns.
    
    Identity Strategy:
        Uses namespaced natural keys for grouping judgements of the same sample.
        A sample's identity is defined by:
        - dataset (from judging_sample.run_info.io_config_name)
        - query external_id (from judging_sample.query.external_id)
        - document external_id (from judging_sample.document.external_id)
        
        This avoids artificial ID generation while handling multi-dataset scenarios
        correctly. Defers database surrogate keys until persistence layer is needed.
    """
    
    def __init__(
        self,
        judgement_reader: JudgementReader,
        aggregated_judgement_writer: AggregatedJudgementWriter,
        strategy: AggregationStrategy,
    ):
        """Initialize aggregation service with port dependencies.
        
        Args:
            judgement_reader: Port for reading LLMJudgement records
            aggregated_judgement_writer: Port for writing AggregatedJudgement records
            strategy: Port for aggregation strategy (e.g., MajorityVoteAdapter)
        """
        self.judgement_reader = judgement_reader
        self.aggregated_judgement_writer = aggregated_judgement_writer
        self.strategy = strategy
        self.logger = get_logger(component="aggregation_service")
    
    @staticmethod
    def _get_sample_identity(judgement: LLMJudgement) -> tuple[str, str, str]:
        """Extract natural composite key for grouping judgements of the same sample.
        
        Uses namespaced natural keys to identify unique query-document pairs:
        - dataset: Identifies which dataset the sample came from (io_config_name)
        - query_id: External query identifier from the original dataset
        - doc_id: External document identifier from the original dataset
        
        This approach:
        - Handles multiple datasets with overlapping IDs correctly
        - Avoids artificial ID generation
        - Is deterministic and reproducible
        - Defers database surrogate keys to persistence layer
        
        Args:
            judgement: LLM judgement to extract identity from

        Returns:
            Tuple of (dataset, query_id, doc_id) serving as natural composite key
        """
        # Extract dataset from embedded query (dataset now flows through pipeline)
        dataset_name = judgement.judging_sample.query.dataset.name
        query_id = judgement.judging_sample.query.external_id
        doc_id = judgement.judging_sample.document.external_id
        return (dataset_name, query_id, doc_id)

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
                f"Found {len(fingerprints)} distinct fingerprints: {fingerprints}. "
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
        1. Reads all judgements via reader port
        2. Groups judgements by natural composite key (dataset, query_id, doc_id)
        3. For each group, applies strategy to get aggregated score
        4. Creates AggregatedJudgement with full judgements + aggregated score
        5. Writes via writer port (streaming)
        6. Tracks statistics (ties, no-votes, etc.)
        
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
        judged_datasets = self.judgement_reader.read(run_names)

        # Validate completion and fingerprint consistency
        self._validate_judged_datasets(judged_datasets, run_names)

        # Log validation
        fingerprints = {dataset.fingerprint for dataset in judged_datasets}
        self.logger.info(
            "validated_judged_datasets",
            num_datasets=len(judged_datasets),
            shared_fingerprint=list(fingerprints)[0][:16] + "..." if fingerprints else "N/A"
        )

        # Extract all judgements from all datasets
        judgements = [
            judgement
            for dataset in judged_datasets
            for judgement in dataset.judgements
        ]

        # Group judgements by natural composite key (dataset, query_id, doc_id)
        grouped: dict[tuple[str, str, str], list[LLMJudgement]] = defaultdict(list)
        for judgement in judgements:
            key = self._get_sample_identity(judgement)
            grouped[key].append(judgement)
        
        # Track statistics
        tie_count = 0
        no_valid_votes_count = 0
        output_count = 0
        
        # Open writer for streaming writes
        with self.aggregated_judgement_writer.open(run_dir) as writer:
            # Process each group
            for (dataset, query_id, doc_id), group_judgements in grouped.items():
                # Apply strategy to get aggregated score
                aggregated_score = self.strategy.aggregate(group_judgements)
                
                # Track statistics
                if aggregated_score.final_relevance_score is None:
                    no_valid_votes_count += 1
                elif "tie" in aggregated_score.final_reasoning.lower():
                    tie_count += 1
                
                # Create aggregated judgement
                aggregated_judgement = AggregatedJudgement(
                    judgements=group_judgements,
                    aggregated_scores=[aggregated_score],  # List to support multiple strategies in future
                )
                
                # Write via writer port
                writer.write_one(aggregated_judgement)
                
                # Log progress
                primary_score = aggregated_judgement.get_primary_aggregated_score()
                final_label = primary_score.final_relevance_score
                confidence = primary_score.final_confidence
                
                self.logger.info(
                    "aggregated_pair",
                    final_label=final_label.label if final_label else "None",
                    confidence=f"{confidence:.2f}" if confidence else "0.00",
                    num_models=len(aggregated_judgement.judgements),
                )
                
                output_count += 1
        
        # Build and finalize summary
        summary_builder.add("input_judgement_count", len(judgements))
        summary_builder.add("unique_pair_count", len(grouped))
        summary_builder.add("output_aggregated_count", output_count)
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
