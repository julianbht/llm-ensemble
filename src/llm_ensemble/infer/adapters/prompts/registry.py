"""Prompt builder registry.

Adapters register themselves using the @prompt_registry.register() decorator.
Adapter modules are imported below to trigger registration.
"""

from llm_ensemble.libs.registry import BaseRegistry


prompt_registry = BaseRegistry()

# Import adapters to trigger registration
from llm_ensemble.infer.adapters.prompts.jinja_prompt_builder import ThomasSimplePromptBuilder  # noqa: F401
