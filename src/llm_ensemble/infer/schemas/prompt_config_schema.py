"""Prompt configuration schema.

Defines the Pydantic schema for prompt template configurations.

Provides convenience methods for instantiating prompt builders and parsers
from their module paths, enforcing the dynamic import pattern.
"""

from __future__ import annotations
from typing import Optional, Any
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class PromptConfig(BaseConfig):
    """Configuration for a prompt template.

    Specifies the template file, builder module, and parser module to use.

    Provides convenience methods for instantiating prompt builders and response parsers
    from their module paths, enforcing the dynamic import pattern.
    """

    description: Optional[str] = Field(None, description="Human-readable description of the prompt")
    prompt_template: str = Field(..., description="Template filename (without .jinja extension)")
    prompt_builder: str = Field(..., description="Builder adapter module name")
    prompt_builder_module_path: str = Field(..., description="Full module path to prompt builder adapter (e.g., 'llm_ensemble.infer.adapters.prompts.jinja_prompt_builder.JinjaPromptBuilder')")
    response_parser: str = Field(..., description="Parser adapter module name")
    response_parser_module_path: str = Field(..., description="Full module path to response parser adapter (e.g., 'llm_ensemble.infer.adapters.parsers.json_response_parser.JsonResponseParser')")

    def get_prompt_builder(self, template: str) -> Any:
        """Instantiate and return the prompt builder adapter.

        Dynamically imports and instantiates the builder class specified
        by prompt_builder_module_path.

        Args:
            template: The prompt template string to pass to the builder

        Returns:
            Instance of the prompt builder adapter

        Raises:
            ImportError: If the builder module path cannot be imported

        Example:
            >>> config = PromptConfig(...)
            >>> template = load_prompt_template(config.prompt_template)
            >>> builder = config.get_prompt_builder(template)
        """
        builder_class = self.instantiate_from_module_path(self.prompt_builder_module_path)
        # Return the class itself, not an instance - factory will instantiate with template
        # Actually, we need to return an instance initialized with the template
        return builder_class(template) if template else builder_class()

    def get_response_parser(self, **kwargs) -> Any:
        """Instantiate and return the response parser adapter.

        Dynamically imports and instantiates the parser class specified
        by response_parser_module_path.

        Args:
            **kwargs: Additional arguments to pass to the parser constructor
                     (e.g., score_field="O")

        Returns:
            Instance of the response parser adapter

        Raises:
            ImportError: If the parser module path cannot be imported

        Example:
            >>> config = PromptConfig(...)
            >>> parser = config.get_response_parser(score_field="O")
        """
        parser_class = self.instantiate_from_module_path(self.response_parser_module_path)
        return parser_class(**kwargs) if kwargs else parser_class()
