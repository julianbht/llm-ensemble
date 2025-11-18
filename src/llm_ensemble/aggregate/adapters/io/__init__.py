"""I/O adapters for aggregate CLI."""

from llm_ensemble.aggregate.adapters.io.sql_judgement_reader import (
    SqlJudgementReader,
)
from llm_ensemble.aggregate.adapters.io.json_aggregated_judgement_writer import (
    JsonAggregatedJudgementWriter,
)

__all__ = [
    "SqlJudgementReader",
    "JsonAggregatedJudgementWriter",
]
