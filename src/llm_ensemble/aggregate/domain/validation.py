"""Domain validation functions for aggregate entities.

Pure domain logic for validating business rules and invariants.
"""

from __future__ import annotations

from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput


def validate_infer_run_outputs_for_aggregation(
    judged_datasets: list[InferRunOutput],
    run_names: list[str],
) -> None:
    """Validate that InferRunOutputs are complete and compatible for aggregation.

    Business rules for valid aggregation inputs:
    1. At least one InferRunOutput must be provided
    2. All InferRunOutputs must have completed successfully (finished=True)
    3. All sample_fingerprints must match (same samples were processed)

    Args:
        judged_datasets: List of InferRunOutput objects loaded by reader
        run_names: Corresponding run names (for error messages)

    Raises:
        ValueError: If validation fails
    """
    if not judged_datasets:
        raise ValueError("No InferRunOutputs found. Cannot aggregate empty list.")

    # Check for incomplete runs (finished=False)
    for dataset, run_name in zip(judged_datasets, run_names):
        if not dataset.finished:
            raise ValueError(
                f"InferRunOutput for run '{run_name}' has finished=False. "
                f"This indicates the run did not complete successfully and may be resumable."
            )

    # Check that all sample_fingerprints match
    sample_fingerprints = {dataset.sample_fingerprint for dataset in judged_datasets}
    if len(sample_fingerprints) > 1:
        raise ValueError(
            f"Cannot aggregate runs with different InferRunOutput sample_fingerprints. "
            f"Found {len(sample_fingerprints)} distinct sample_fingerprints. "
            f"This means the runs processed different sets of samples."
        )
