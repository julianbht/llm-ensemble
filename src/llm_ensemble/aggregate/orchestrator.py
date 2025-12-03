"""Orchestrator for the aggregate CLI.

This module handles infrastructure concerns for the aggregation pipeline:
- Loading configurations
- Setting up run directories and logging
- Building manifests with git info and execution parameters
- Instantiating adapters (strategy, reader, writer)
- Delegating business logic to domain service
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.aggregate.schemas.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.registry import AggregationStrategyBuilder
from llm_ensemble.aggregate.domain import AggregationService
from llm_ensemble.libs.schemas import IOConfig, LoggingConfig
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.runtime.run_summary_builder import write_standalone_summary
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.runtime.git_utils import get_git_info
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.logging import configure_logger


def run_aggregation(
    aggregation_strategy_name: str,
    io_config: IOConfig,
    logging_config: LoggingConfig,
    input_run_names: list[str],
    io_config_name: str,
    run_name: Optional[str] = None,
    official: bool = False,
    notes: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    """Run ensemble aggregation with full provenance.

    Infrastructure orchestration that coordinates:
    - Setting up run directories and logging
    - Building manifest with git info and execution parameters
    - Instantiating adapters from builder
    - Running aggregation and writing output

    Args:
        aggregation_strategy_name: Name of aggregation strategy (e.g., 'majority_vote')
        io_config: I/O configuration (reader/writer adapters)
        logging_config: Logging configuration
        input_run_names: List of infer run identifiers to read judgements from
        io_config_name: Name of the I/O config file
        run_name: Custom run ID (auto-generates if not provided)
        official: Mark as official run
        notes: Notes about this run
        tag: Tag name for easy reference by downstream CLIs

    Raises:
        FileNotFoundError: If any run directory doesn't exist
        ValueError: If strategy not found
    """

    # Generate or use provided run_name
    if run_name is None:
        run_name = generate_run_name([
            aggregation_strategy_name,
            io_config.name_hint,
        ])

    # Get run directory path and create it
    run_dir = PathManager.get_run_dir(
        cli_name="aggregate",
        run_name=run_name,
        official=official
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Register tag if provided
    if tag:
        TagManager.register_tag(tag, "aggregate", run_name)


    # Get git info for reproducibility
    git_info = get_git_info()

    # Create immutable run info
    run_info = AggregateRunInfo(
        run_name=run_name,
        run_type=RunType.OFFICIAL if official else RunType.TEST,
        notes=notes,
        git_sha=git_info["git_sha"],
        git_clean=git_info["git_clean"],
        git_branch=git_info["git_branch"],
        aggregation_strategy_name=aggregation_strategy_name,
        io_config_name=io_config_name,
        io_config=io_config,
        input_run_names=input_run_names,
    )
    
    # Set up log file path if saving logs
    log_file_path = run_dir / "run.log" if logging_config.save_logs else None
    
    # Initialize logger
    logger = configure_logger(
        cli_name="aggregate",
        run_name=run_name,
        run_type=run_info.run_type,
        pretty_print=logging_config.pretty_print,
        save_logs=logging_config.save_logs,
        log_file_path=log_file_path,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
    )

    # Instantiate adapters from builder and config
    aggregation_strategy_adapter = AggregationStrategyBuilder.build(aggregation_strategy_name)
    reader = io_config.get_reader()
    writer = io_config.get_writer()

    logger.info(
        "starting_aggregation",
        strategy=aggregation_strategy_name,
        io_format=io_config_name,
        input_run_names=input_run_names,
    )
    logger.info("run_directory", path=str(run_dir))

    # Create domain service
    service = AggregationService(
        judgement_reader=reader,
        aggregated_judgement_writer=writer,
        aggregation_strategy_adapter=aggregation_strategy_adapter,
    )
    
    # Run aggregation pipeline
    try:
        summary = service.run_aggregation(
            run_names=input_run_names,
            run_info=run_info,
            run_dir=run_dir,
        )
        
        logger.info(
            "aggregation_complete",
            input_judgements=summary.input_judgement_count,
            unique_pairs=summary.unique_pair_count,
            output_aggregated=summary.output_aggregated_count,
        )
        
        # Write standalone summary.json
        write_standalone_summary(summary, run_dir)
        logger.info("summary_written", path=str(run_dir / "summary.json"))
        
    except Exception as e:
        logger.error("aggregation_failed", error=str(e))
        raise
    
    # Log warnings summary if any
    if summary.warnings_summary:
        logger.info(
            "warnings_collected",
            **summary.warnings_summary
        )
    
    # Log where logs were saved if enabled
    if logging_config.save_logs:
        logger.info("logs_saved", path=str(run_dir / "run.log"))
