"""Response parser registry.

Adapters register themselves using the @parser_registry.register() decorator.
Import adapter modules to trigger registration (done in infer_cli.py).
"""

from llm_ensemble.libs.registry import BaseRegistry


parser_registry = BaseRegistry()
