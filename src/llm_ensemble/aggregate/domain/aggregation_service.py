"""Domain service for aggregation pipeline.

This module contains pure business logic for orchestrating ensemble aggregation.
It depends only on port abstractions and has no knowledge of infrastructure
details (file formats, I/O).
"""

from __future__ import annotations
from typing import Callable, Optional
from collections import defaultdict

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas import AggregatedJudgement
from llm_ensemble.aggregate.schemas.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.schemas.aggregate_run_summary import AggregateRunSummary
from llm_ensemble.aggregate.ports import (
    AggregationStrategy,
    JudgementReader,
    AggregatedJudgementWriter,
)
from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder


class AggregationService:
    """Domain service for coordinating ensemble aggregation pipeline.
    
    Pure business logic that orchestrates:
    - Reading judgements via JudgementReader port
    - Grouping judgements by (query_id, docid)
    - Applying aggregation strategy to each group
    - Writing aggregated judgements via AggregatedJudgementWriter port
    - Tracking statistics (ties, no-votes)
    
    Depends only on port abstractions, enabling complete independence from
    infrastructure concerns.
    
    Example:
        >>> reader = JsonJudgementReader()
        >>> writer = JsonAggregatedJudgementWriter()
        >>> strategy = MajorityVoteAdapter()
        >>> service = AggregationService(reader, writer, strategy)
        >>> summary = service.run_aggregation(
        ...     input_paths=[Path("judgements.json")],
        ...     run_info=run_info,
        ...     run_dir=run_dir,
        ... )
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
    
    def run_aggregation(
        self,
        input_paths: list,
        run_info: AggregateRunInfo,
        run_dir,
        on_aggregated: Optional[Callable[[AggregatedJudgement], None]] = None,
    ) -> AggregateRunSummary:
        """Execute the aggregation pipeline.
        
        Pure business logic that:
        1. Reads all judgements via reader port
        2. Groups judgements by (query_id, docid)
        3. For each group, applies strategy to get aggregated score
        4. Creates AggregatedJudgement with full judgements + aggregated score
        5. Writes via writer port (streaming)
        6. Tracks statistics (ties, no-votes, etc.)
        
        Args:
            input_paths: List of paths to files containing LLMJudgement records
            run_info: Immutable runtime context (attached to summary)
            run_dir: Run directory for output
            on_aggregated: Optional callback invoked for each aggregated judgement
            
        Returns:
            AggregateRunSummary with statistics
        """
        # Initialize summary builder with run_info
        builder = RunSummaryBuilder(run_info)
        builder.set_start_time()
        
        # Read all judgements via reader port
        judgements = self.judgement_reader.read(input_paths)
        
        # Group judgements by (query_id, docid)
        grouped: dict[tuple[str, str], list[LLMJudgement]] = defaultdict(list)
        for judgement in judgements:
            query_id = judgement.judging_sample.query.query_id
            docid = judgement.judging_sample.document.docid
            key = (query_id, docid)
            grouped[key].append(judgement)
        
        # Track statistics
        tie_count = 0
        no_valid_votes_count = 0
        output_count = 0
        
        # Open writer for streaming writes
        with self.aggregated_judgement_writer.open(run_dir) as writer:
            # Process each group
            for (query_id, docid), group_judgements in grouped.items():
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
                
                # Invoke callback for progress tracking
                if on_aggregated:
                    on_aggregated(aggregated_judgement)
                
                output_count += 1
        
        # Build and finalize summary
        builder.add("input_judgement_count", len(judgements))
        builder.add("unique_pair_count", len(grouped))
        builder.add("output_aggregated_count", output_count)
        builder.add("tie_count", tie_count)
        builder.add("no_valid_votes_count", no_valid_votes_count)
        
        # Optional: add warnings summary
        warnings_summary = {}
        if tie_count > 0:
            warnings_summary["tie"] = tie_count
        if no_valid_votes_count > 0:
            warnings_summary["no_valid_votes"] = no_valid_votes_count
        if warnings_summary:
            builder.add("warnings_summary", warnings_summary)
        
        return builder.finalize(AggregateRunSummary)
