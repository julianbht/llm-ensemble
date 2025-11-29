"""Port interfaces for aggregate CLI.

Defines abstract contracts that adapters must implement.
This enables dependency inversion - domain logic depends on abstractions,
not concrete implementations.
"""

from llm_ensemble.aggregate.ports.aggregation_strategy import AggregationStrategyPort
from llm_ensemble.aggregate.ports.judgement_reader import JudgementReader
from llm_ensemble.aggregate.ports.aggregated_judgement_writer import AggregatedJudgementWriter

__all__ = [
    "AggregationStrategyPort",
    "JudgementReader",
    "AggregatedJudgementWriter",
]
