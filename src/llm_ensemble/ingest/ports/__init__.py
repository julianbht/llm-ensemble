"""Ports for the ingest CLI.

This package defines abstract interfaces (ports) that the domain layer
depends on. Concrete implementations (adapters) are in the adapters/ package.
"""

from llm_ensemble.ingest.ports.dataset_reader import DatasetReader
from llm_ensemble.ingest.ports.dataset_writer import DatasetWriter

__all__ = [
    "DatasetReader",
    "DatasetWriter",
]
