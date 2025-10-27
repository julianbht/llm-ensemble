"""Adapters for the ingest CLI.

This package contains concrete implementations of ports (example readers,
example writers) and factory functions for instantiating them.
"""

from llm_ensemble.ingest.adapters.io_factory import get_example_reader, get_example_writer

__all__ = ["get_example_reader", "get_example_writer"]
