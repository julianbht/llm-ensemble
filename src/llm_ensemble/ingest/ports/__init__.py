"""Ports for the ingest CLI.

This package defines abstract interfaces (ports) that the domain layer
depends on. Concrete implementations (adapters) are in the adapters/ package.
"""

from llm_ensemble.ingest.ports.dataset_adapter import DatasetAdapter
from llm_ensemble.ingest.ports.example_writer import ExampleWriter

__all__ = ["DatasetAdapter", "ExampleWriter"]
