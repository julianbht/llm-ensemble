"""I/O adapters for reading examples and writing judgements."""

from llm_ensemble.infer.adapters.io.fully_populated_json_reader import FullyPopulatedJsonReader
from llm_ensemble.infer.adapters.io.fully_populated_json_writer import FullyPopulatedJsonWriter

__all__ = [
    "FullyPopulatedJsonReader",
    "FullyPopulatedJsonWriter",
]
