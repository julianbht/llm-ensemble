"""Prompt builder registry.

Adapters register themselves using the @prompt_registry.register() decorator.
Import adapter modules to trigger registration (done in infer_cli.py).
"""

from llm_ensemble.libs.registry import BaseRegistry


prompt_registry = BaseRegistry()
