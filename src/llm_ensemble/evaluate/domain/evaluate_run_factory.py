"""Factory for creating EvaluateRun entities.

Domain Layer - Factory Pattern

Creates EvaluateRun aggregate root from domain entities and primitive values,
assembling the complete record for persistence.

This factory belongs in the domain layer because it only depends on
domain entities and performs pure assembly logic. The application layer
is responsible for providing these values.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from llm_ensemble.evaluate.domain.entities.evaluate_run import EvaluateRun
from llm_ensemble.evaluate.domain.entities.evaluate_run_config import EvaluateRunConfig
from llm_ensemble.evaluate.domain.entities.metric_result import MetricResult
from llm_ensemble.libs.runtime.run_info import RunType


class EvaluateRunFactory:
    """Factory for creating EvaluateRun aggregate root from domain entities.

    Domain layer factory - pure assembly logic with no adapter dependencies.
    """

    @staticmethod
    def create(
        io_config_name: str,
        input_run_name: str,
        metric_names: list[str],
        run_name: str,
        run_type: RunType,
        metric_results: list[MetricResult],
        evaluated_run_type: str,
        evaluated_sample_count: int,
        start_time: datetime,
        end_time: datetime,
        notes: Optional[str],
    ) -> EvaluateRun:
        """Create EvaluateRun aggregate root from domain entities and primitive values.

        Args:
            io_config_name: Name of the I/O configuration used
            input_run_name: Run identifier to evaluate (infer or aggregate run)
            metric_names: List of metric names computed
            run_name: Run identifier
            run_type: Type of run (OFFICIAL or TEST)
            metric_results: List of computed metric results
            evaluated_run_type: Type of run that was evaluated ('infer' or 'aggregate')
            evaluated_sample_count: Number of samples in the evaluated dataset
            start_time: When the run started
            end_time: When the run completed
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            Assembled EvaluateRun aggregate root
        """
        # Build EvaluateRunConfig entity
        run_config = EvaluateRunConfig(
            io_config_name=io_config_name,
            input_run_name=input_run_name,
            metric_names=metric_names,
        )

        # Assemble complete aggregate root
        return EvaluateRun(
            run_name=run_name,
            run_type=run_type,
            evaluate_run_config=run_config,
            metric_results=metric_results,
            evaluated_run_type=evaluated_run_type,
            evaluated_sample_count=evaluated_sample_count,
            start_time=start_time,
            end_time=end_time,
            notes=notes,
        )
