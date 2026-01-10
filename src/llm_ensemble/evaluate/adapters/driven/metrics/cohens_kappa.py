"""Cohen's Kappa metric adapter.

Driven Adapter - Metrics (Hexagonal Architecture)

Implements ForComputingMetrics port using scikit-learn's cohen_kappa_score.

Cohen's Kappa measures inter-rater agreement for categorical items, accounting
for agreement occurring by chance. Values range from -1 to 1:
- 1 = perfect agreement
- 0 = no agreement beyond chance
- < 0 = less agreement than expected by chance

Reference: Cohen, J. (1960). A coefficient of agreement for nominal scales.
"""

from __future__ import annotations
from typing import Optional

from sklearn.metrics import cohen_kappa_score

from llm_ensemble.evaluate.application.ports.driven.for_computing_metrics import ForComputingMetrics
from llm_ensemble.evaluate.domain.entities.metric_result import MetricResult
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class CohensKappaAdapter(ForComputingMetrics):
    """Cohen's Kappa metric adapter using scikit-learn.

    Implements ForComputingMetrics port for computing inter-rater agreement.
    """

    def compute(
        self,
        ground_truth: list[RelevanceScore],
        predictions: list[Optional[RelevanceScore]]
    ) -> MetricResult:
        """Compute Cohen's Kappa coefficient.

        Args:
            ground_truth: List of ground truth relevance labels
            predictions: List of predicted relevance labels (None if parse failed)

        Returns:
            MetricResult entity with Cohen's Kappa value and interpretation

        Raises:
            ValueError: If inputs have different lengths or are empty
        """
        if len(ground_truth) != len(predictions):
            raise ValueError(
                f"Ground truth and predictions must have same length "
                f"(got {len(ground_truth)} vs {len(predictions)})"
            )

        if len(ground_truth) == 0:
            raise ValueError("Cannot compute Cohen's Kappa on empty input")

        # Compute Cohen's Kappa using scikit-learn
        kappa_value = cohen_kappa_score(ground_truth, predictions)

        # Interpretation guide (Landis & Koch, 1977)
        interpretation = self._interpret_kappa(kappa_value)

        return MetricResult(
            name="cohens_kappa",
            value=kappa_value,
            sample_size=len(ground_truth),
            interpretation=interpretation,
            description="Cohen's Kappa coefficient measuring inter-rater agreement",
            min_value=-1.0,
            max_value=1.0,
            higher_is_better=True,
        )

    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Interpret Cohen's Kappa value using Landis & Koch (1977) scale.

        Args:
            kappa: Cohen's Kappa coefficient

        Returns:
            Interpretation string
        """
        if kappa < 0:
            return "poor (less than chance agreement)"
        elif kappa < 0.20:
            return "slight"
        elif kappa < 0.40:
            return "fair"
        elif kappa < 0.60:
            return "moderate"
        elif kappa < 0.80:
            return "substantial"
        else:
            return "almost perfect"
