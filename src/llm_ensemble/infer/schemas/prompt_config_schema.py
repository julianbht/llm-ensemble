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
    builder_module: str = Field(..., description="Full Python module path to prompt builder (e.g., 'llm_ensemble.infer.adapters.prompts.jinja_prompt_builder')")
    builder_class: str = Field(..., description="Prompt builder class name in UpperCamelCase (e.g., 'JinjaPromptBuilder')")
    parser_module: str = Field(..., description="Full Python module path to response parser (e.g., 'llm_ensemble.infer.adapters.parsers.json_response_parser')")
    parser_class: str = Field(..., description="Response parser class name in UpperCamelCase (e.g., 'JsonResponseParser')")

    def get_prompt_builder(self, template: str) -> Any:
        """Instantiate and return the prompt builder adapter.

        Dynamically imports the builder module and instantiates the builder class.

        Args:
            template: The prompt template string to pass to the builder

        Returns:
            Instance of the prompt builder adapter

        Raises:
            ImportError: If the builder module cannot be imported
            AttributeError: If the builder class doesn't exist in the module

        Example:
            >>> config = PromptConfig(...)
            >>> template = load_prompt_template(config.prompt_template)
            >>> builder = config.get_prompt_builder(template)
        """
        return self._instantiate_adapter(self.builder_module, self.builder_class, template=template)

    def get_response_parser(self, score_field: str = "O") -> Any:
        """Instantiate and return the response parser adapter.

        Dynamically imports the parser module and instantiates the parser class.

        Args:
            score_field: Field name to extract score from (default: "O")

        Returns:
            Instance of the response parser adapter

        Raises:
            ImportError: If the parser module cannot be imported
            AttributeError: If the parser class doesn't exist in the module

        Example:
            >>> config = PromptConfig(...)
            >>> parser = config.get_response_parser(score_field="O")
        """
        return self._instantiate_adapter(self.parser_module, self.parser_class, score_field=score_field)
