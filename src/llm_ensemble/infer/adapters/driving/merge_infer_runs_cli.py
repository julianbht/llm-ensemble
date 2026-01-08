"""Merge Infer Runs CLI - Utility for merging continuation runs into incomplete runs

This utility script merges a continuation run into an incomplete (interrupted) run.
Used for resuming long-running inference jobs that were interrupted.

Usage:
    merge-infer-runs --source <continuation_run_name> --target <incomplete_run_name> [--dry-run]

Example:
    merge-infer-runs --source gpt-5-1-continuation --target gpt-5-1-all-samples-base-reference

What it does:
1. Validates that target is incomplete (finished=False) and source is complete (finished=True)
2. Validates no overlapping samples (would violate unique constraint)
3. Validates compatible configurations (same model, prompt, provider)
4. Migrates all judgements from source → target
5. Recomputes target's sample fingerprint
6. Marks target as finished=True
7. Deletes source run records and artifacts

All operations are performed in a single database transaction for safety.
"""

from __future__ import annotations
import sys
import shutil
import typer
from sqlalchemy.orm import Session

# Load runtime env configuration (DATABASE_URL, API keys, etc.)
from llm_ensemble.libs.runtime.env import load_runtime_config
load_runtime_config()

from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    InferRunOutputORM,
    InferRunConfigORM,
    LLMJudgementORM,
)
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import get_session
from llm_ensemble.libs.db.uuid_helpers import compute_judged_dataset_fingerprint
from llm_ensemble.libs.logging.structlog_logger import get_logger
from llm_ensemble.libs.runtime.path_manager import PathManager

app = typer.Typer(
    add_completion=True,
    help="Merge continuation run into incomplete run",
    pretty_exceptions_enable=False,
)

logger = get_logger(component="merge_infer_runs")


def validate_runs(
    session: Session,
    source_run_name: str,
    target_run_name: str,
) -> tuple[InferRunORM, InferRunORM]:
    """Validate source and target runs exist and are in correct states.

    Args:
        session: Database session
        source_run_name: Name of continuation run (source)
        target_run_name: Name of incomplete run (target)

    Returns:
        Tuple of (source_run, target_run)

    Raises:
        ValueError: If validation fails
    """
    # Fetch runs
    source_run = session.query(InferRunORM).filter_by(run_name=source_run_name).first()
    target_run = session.query(InferRunORM).filter_by(run_name=target_run_name).first()

    if not source_run:
        raise ValueError(f"Source run '{source_run_name}' not found in database")
    if not target_run:
        raise ValueError(f"Target run '{target_run_name}' not found in database")

    # Fetch outputs
    source_output = session.query(InferRunOutputORM).filter_by(id=source_run.id).first()
    target_output = session.query(InferRunOutputORM).filter_by(id=target_run.id).first()

    if not source_output:
        raise ValueError(f"Source run '{source_run_name}' has no output (corrupt data)")
    if not target_output:
        raise ValueError(f"Target run '{target_run_name}' has no output (corrupt data)")

    # Validate states
    if not source_output.finished:
        raise ValueError(
            f"Source run '{source_run_name}' is not finished (finished=False). "
            "Only complete runs can be merged."
        )
    if target_output.finished:
        raise ValueError(
            f"Target run '{target_run_name}' is already finished (finished=True). "
            "Cannot merge into a completed run."
        )

    logger.info(
        "validation_passed",
        source=source_run_name,
        target=target_run_name,
        source_finished=source_output.finished,
        target_finished=target_output.finished,
    )

    return source_run, target_run


def validate_no_overlapping_samples(
    session: Session,
    source_run: InferRunORM,
    target_run: InferRunORM,
) -> None:
    """Validate that source and target have no overlapping samples.

    Args:
        session: Database session
        source_run: Source run ORM
        target_run: Target run ORM

    Raises:
        ValueError: If there are overlapping samples
    """
    # Get all sample IDs from both runs
    source_sample_ids = set(
        session.query(LLMJudgementORM.normalized_dataset_judging_sample_id)
        .filter_by(infer_run_output_id=source_run.id)
        .all()
    )
    target_sample_ids = set(
        session.query(LLMJudgementORM.normalized_dataset_judging_sample_id)
        .filter_by(infer_run_output_id=target_run.id)
        .all()
    )

    # Flatten tuples
    source_sample_ids = {sid[0] for sid in source_sample_ids}
    target_sample_ids = {sid[0] for sid in target_sample_ids}

    overlap = source_sample_ids & target_sample_ids

    if overlap:
        raise ValueError(
            f"Source and target runs have {len(overlap)} overlapping samples. "
            "This would violate the unique constraint on (infer_run_output_id, normalized_dataset_judging_sample_id). "
            f"Overlapping sample IDs: {list(overlap)[:5]}..."
        )

    logger.info(
        "no_overlapping_samples",
        source_count=len(source_sample_ids),
        target_count=len(target_sample_ids),
    )


def validate_compatible_configs(
    session: Session,
    source_run: InferRunORM,
    target_run: InferRunORM,
) -> None:
    """Validate that source and target have compatible configurations.

    Args:
        session: Database session
        source_run: Source run ORM
        target_run: Target run ORM

    Raises:
        ValueError: If configurations are incompatible
    """
    source_config = session.query(InferRunConfigORM).filter_by(id=source_run.infer_run_config_id).first()
    target_config = session.query(InferRunConfigORM).filter_by(id=target_run.infer_run_config_id).first()

    if not source_config or not target_config:
        raise ValueError("Failed to fetch run configurations")

    # Check critical config fields
    if source_config.model_config_id != target_config.model_config_id:
        raise ValueError(
            "Source and target runs have different model configurations. "
            "Merge is only supported for runs with identical configs."
        )

    if source_config.provider_id != target_config.provider_id:
        raise ValueError("Source and target runs have different providers")

    if source_config.prompt_template_id != target_config.prompt_template_id:
        raise ValueError("Source and target runs have different prompt templates")

    if source_config.input_run_name != target_config.input_run_name:
        raise ValueError("Source and target runs have different input sources")

    logger.info(
        "compatible_configs",
        model_config_id=str(source_config.model_config_id),
        provider_id=str(source_config.provider_id),
        prompt_template_id=str(source_config.prompt_template_id),
    )


def perform_merge(
    session: Session,
    source_run: InferRunORM,
    target_run: InferRunORM,
    delete_artifacts: bool = False,
) -> dict[str, int]:
    """Perform the actual merge operation.

    Args:
        session: Database session
        source_run: Source run ORM
        target_run: Target run ORM
        delete_artifacts: Whether to delete source run artifacts

    Returns:
        Dict with merge statistics
    """
    # Count judgements before merge
    source_judgement_count = session.query(LLMJudgementORM).filter_by(
        infer_run_output_id=source_run.id
    ).count()
    target_judgement_count_before = session.query(LLMJudgementORM).filter_by(
        infer_run_output_id=target_run.id
    ).count()

    logger.info(
        "merge_starting",
        source_judgements=source_judgement_count,
        target_judgements_before=target_judgement_count_before,
    )

    # Step 1: Migrate all source judgements to target
    session.query(LLMJudgementORM).filter_by(
        infer_run_output_id=source_run.id
    ).update({"infer_run_output_id": target_run.id})

    logger.info("judgements_migrated", count=source_judgement_count)

    # Step 2: Recompute target's sample fingerprint
    all_target_sample_ids = [
        row[0] for row in session.query(LLMJudgementORM.normalized_dataset_judging_sample_id)
        .filter_by(infer_run_output_id=target_run.id)
        .all()
    ]
    new_fingerprint = compute_judged_dataset_fingerprint(all_target_sample_ids)

    target_output = session.query(InferRunOutputORM).filter_by(id=target_run.id).first()
    assert target_output is not None  # Already validated in validate_runs
    setattr(target_output, "sample_fingerprint", new_fingerprint)
    setattr(target_output, "finished", True)

    logger.info(
        "target_finalized",
        sample_count=len(all_target_sample_ids),
        fingerprint=new_fingerprint[:16] + "...",
    )

    # Step 3: Recompute target's end_time by adding source run duration
    # This ensures accurate timing by excluding the gap between runs
    # Formula: new_end_time = target_end_time + (source_end_time - source_start_time)
    source_duration = source_run.end_time - source_run.start_time
    new_end_time = target_run.end_time + source_duration
    setattr(target_run, "end_time", new_end_time)

    logger.info(
        "target_end_time_recomputed",
        original_end_time=str(target_run.end_time),
        source_duration_seconds=source_duration.total_seconds(),
        new_end_time=str(new_end_time),
    )

    # Step 4: Delete source run (CASCADE will delete source output)
    session.delete(source_run)
    logger.info("source_run_deleted", run_name=source_run.run_name)

    # Step 5: Optionally delete source run artifacts
    if delete_artifacts:
        try:
            source_run_dir = PathManager.resolve_run_dir("infer", str(source_run.run_name))
            if source_run_dir.exists():
                shutil.rmtree(source_run_dir)
                logger.info("source_artifacts_deleted", path=str(source_run_dir))
            else:
                logger.warning("source_artifacts_not_found", path=str(source_run_dir))
        except FileNotFoundError:
            logger.warning("source_artifacts_not_found", run_name=str(source_run.run_name))
    else:
        logger.info("source_artifacts_kept", run_name=str(source_run.run_name))

    # Verify final state
    target_judgement_count_after = session.query(LLMJudgementORM).filter_by(
        infer_run_output_id=target_run.id
    ).count()

    return {
        "source_judgements": source_judgement_count,
        "target_judgements_before": target_judgement_count_before,
        "target_judgements_after": target_judgement_count_after,
        "total_judgements": target_judgement_count_after,
    }


@app.command("merge")
def merge(
    source: str = typer.Option(..., "--source", help="Name of continuation run to merge FROM"),
    target: str = typer.Option(..., "--target", help="Name of incomplete run to merge INTO"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without making changes"),
    delete_artifacts: bool = typer.Option(False, "--delete-artifacts", help="Delete source run artifacts after merge"),
):
    """Merge continuation run into incomplete run.

    This utility merges judgements from a continuation run into an incomplete
    (interrupted) run, allowing you to resume long-running inference jobs.
    """
    logger.info("merge_started", source=source, target=target, dry_run=dry_run)

    engine = get_engine()
    session = get_session(engine)

    try:
        # Validation phase
        logger.info("validation_phase_starting")
        source_run, target_run = validate_runs(session, source, target)
        validate_no_overlapping_samples(session, source_run, target_run)
        validate_compatible_configs(session, source_run, target_run)
        logger.info("validation_phase_complete")

        if dry_run:
            logger.info("dry_run_mode", message="Validation passed. No changes made.")
            session.close()
            return

        # Merge phase
        logger.info("merge_phase_starting")
        stats = perform_merge(
            session,
            source_run,
            target_run,
            delete_artifacts=delete_artifacts,
        )
        session.commit()
        logger.info("merge_phase_complete", **stats)

        # Success summary
        typer.echo("\n" + "=" * 60)
        typer.echo("MERGE SUCCESSFUL")
        typer.echo("=" * 60)
        typer.echo(f"Source run '{source}' merged into '{target}'")
        typer.echo(f"Total judgements in target: {stats['total_judgements']}")
        typer.echo(f"  - Original: {stats['target_judgements_before']}")
        typer.echo(f"  - Added: {stats['source_judgements']}")
        typer.echo("=" * 60 + "\n")

    except ValueError as e:
        logger.error("validation_failed", error=str(e))
        typer.echo(f"\nERROR: {e}", err=True)
        session.rollback()
        sys.exit(1)
    except Exception as e:
        logger.error("merge_failed", error=str(e), error_type=type(e).__name__)
        typer.echo(f"\nUNEXPECTED ERROR: {e}", err=True)
        session.rollback()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    app()
