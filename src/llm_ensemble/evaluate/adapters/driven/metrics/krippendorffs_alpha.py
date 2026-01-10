"""Krippendorff's Alpha metric adapter.

Driven Adapter - Metrics (Hexagonal Architecture)

Implements ForComputingMetrics port using fast-krippendorff package.

Krippendorff's Alpha measures inter-rater reliability for ordinal data with
support for missing values. Values range from -1 to 1:
- 1 = perfect agreement
- 0 = no agreement beyond chance
- < 0 = systematic disagreement

Reference: Krippendorff, K. (2004). Reliability in Content Analysis.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import krippendorff

from llm_ensemble.evaluate.application.ports.driven.for_computing_metrics import ForComputingMetrics
from llm_ensemble.evaluate.domain.entities.metric_result import MetricResult
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class KrippendorffsAlphaAdapter(ForComputingMetrics):
    """Krippendorff's Alpha metric adapter using fast-krippendorff.

    Implements ForComputingMetrics port for computing inter-rater reliability
    with support for ordinal scales and missing data.
    """

    def compute(
        self,
        ground_truth: list[RelevanceScore],
        predictions: list[Optional[RelevanceScore]]
    ) -> MetricResult:
        """Compute Krippendorff's Alpha coefficient for ordinal data.

        Args:
            ground_truth: List of ground truth relevance labels
            predictions: List of predicted relevance labels (None if parse failed)

        Returns:
            MetricResult entity with Krippendorff's Alpha value and interpretation

        Raises:
            ValueError: If inputs have different lengths or are empty
        """
        if len(ground_truth) != len(predictions):
            raise ValueError(
                f"Ground truth and predictions must have same length "
                f"(got {len(ground_truth)} vs {len(predictions)})"
            )

        if len(ground_truth) == 0:
            raise ValueError("Cannot compute Krippendorff's Alpha on empty input")

        # Check if there are any valid predictions
        valid_count = sum(1 for pred in predictions if pred is not None)
        missing_count = len(predictions) - valid_count

        if valid_count == 0:
            raise ValueError(
                "Cannot compute Krippendorff's Alpha: all predictions are None (no valid data)"
            )

        # Assert data quality: no more than 5 missing labels
        if missing_count > 5:
            raise ValueError(
                f"Too many missing predictions ({missing_count}). "
                f"Maximum allowed: 5. Data may not be usable."
            )

        # Format data for krippendorff package (2 raters x N units matrix)
        reliability_data = self._format_reliability_data(ground_truth, predictions)

        # Compute Krippendorff's Alpha using ordinal scale
        try:
            alpha_value = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement="ordinal",
                value_domain=[0, 1, 2, 3]  # RelevanceScore values (IRRELEVANT to PERFECTLY_RELEVANT)
            )
        except ValueError as e:
            # Handle edge case: all predictions are None (insufficient data)
            if "at least one unit with values assigned by at least two coders" in str(e):
                alpha_value = np.nan
            else:
                raise

        # Interpretation guide (Krippendorff, 2004)
        interpretation = self._interpret_alpha(alpha_value)

        # Count non-missing samples for sample_size
        sample_size = self._count_valid_pairs(ground_truth, predictions)

        return MetricResult(
            name="krippendorffs_alpha",
            value=alpha_value,
            sample_size=sample_size,
            interpretation=interpretation,
            description="Krippendorff's Alpha coefficient measuring inter-rater reliability (ordinal scale)",
            min_value=-1.0,
            max_value=1.0,
            higher_is_better=True,
        )

    @staticmethod
    def _format_reliability_data(
        ground_truth: list[RelevanceScore],
        predictions: list[Optional[RelevanceScore]]
    ) -> np.ndarray:
        """Format data for krippendorff package (2 raters x N units).

        The krippendorff package expects:
        - reliability_data: shape (M, N) where M=raters, N=units
        - Missing values as np.nan
        - Row 0 = rater 1 (ground truth)
        - Row 1 = rater 2 (predictions)

        Args:
            ground_truth: Ground truth labels (always present)
            predictions: Predicted labels (None if parse failed)

        Returns:
            2D numpy array with shape (2, N) suitable for krippendorff.alpha()
        """
        # Convert ground_truth (RelevanceScore enum) to numeric values
        ground_truth_array = np.array([int(score) for score in ground_truth], dtype=np.float64)

        # Convert predictions, mapping None to np.nan
        predictions_array = np.array([
            float(score) if score is not None else np.nan
            for score in predictions
        ], dtype=np.float64)

        # Stack as 2 x N matrix (2 raters, N units)
        reliability_data = np.vstack([ground_truth_array, predictions_array])

        return reliability_data

    @staticmethod
    def _count_valid_pairs(
        ground_truth: list[RelevanceScore],
        predictions: list[Optional[RelevanceScore]]
    ) -> int:
        """Count number of valid (non-None) prediction pairs.

        Sample size reflects actual pairs used in computation.

        Args:
            ground_truth: Ground truth labels
            predictions: Predicted labels (None if parse failed)

        Returns:
            Count of valid pairs
        """
        return sum(1 for pred in predictions if pred is not None)

    @staticmethod
    def _interpret_alpha(alpha: float) -> str:
        """Interpret Krippendorff's Alpha value using standard thresholds.

        Based on Krippendorff (2004): "It is customary to require α ≥ .800.
        Where tentative conclusions are still acceptable, α ≥ .667 is the
        lowest conceivable limit."

        Args:
            alpha: Krippendorff's Alpha coefficient

        Returns:
            Interpretation string
        """
        if np.isnan(alpha):
            return "undefined (insufficient data)"
        elif alpha < 0:
            return "poor (systematic disagreement)"
        elif alpha < 0.67:
            return "insufficient (unreliable)"
        elif alpha < 0.80:
            return "tentative (use caution)"
        else:
            return "strong (reliable)"
