"""I/O adapters for aggregate CLI."""

from llm_ensemble.aggregate.adapters.io.json_judgement_reader import (
    JsonJudgementReader,
)
from llm_ensemble.aggregate.adapters.io.json_aggregated_judgement_writer import (
    JsonAggregatedJudgementWriter,
)

__all__ = [
    "JsonJudgementReader",
    "JsonAggregatedJudgementWriter",
]
