"""Shared configuration loading utilities.

This module provides generic YAML config loaders used across all CLIs.
"""

from llm_ensemble.libs.config.yaml_config_loader import load_yaml_config

__all__ = ["load_yaml_config"]
