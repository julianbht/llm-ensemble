"""Aggregate pipeline - combine judgements using ensemble strategies.

Import strategies to ensure they register themselves with the registry.
"""

from llm_ensemble.aggregate.adapters import strategies  # noqa: F401
from llm_ensemble.aggregate.domain.aggregation_service import AggregationService

__all__ = ["AggregationService"]
