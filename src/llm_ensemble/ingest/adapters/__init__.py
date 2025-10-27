"""Adapters for the ingest CLI.

This package contains concrete implementations of ports (dataset adapters,
example writers) and factory functions for instantiating them.
"""

from llm_ensemble.ingest.adapters.dataset_adapter_factory import get_dataset_adapter
from llm_ensemble.ingest.adapters.example_writer_factory import get_example_writer

__all__ = ["get_dataset_adapter", "get_example_writer"]
