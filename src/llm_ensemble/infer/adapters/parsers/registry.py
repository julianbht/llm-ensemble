"""Response parser registry.

Adapters register themselves using the @parser_registry.register() decorator.
Adapter modules are imported below to trigger registration.
"""

from llm_ensemble.libs.registry import BaseRegistry


parser_registry = BaseRegistry()

# Import adapters to trigger registration
from llm_ensemble.infer.adapters.parsers.thomas_simple_parser import ThomasSimpleParser  # noqa: F401
