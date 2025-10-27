"""Adapters for the ingest CLI.

This package contains concrete implementations of ports (sample readers,
dataset writers) and factory functions for instantiating them.
"""

from llm_ensemble.ingest.adapters.io_factory import get_sample_reader, get_dataset_writer

__all__ = ["get_sample_reader", "get_dataset_writer"]
