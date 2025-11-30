"""Prompt-parser configuration schema.

Complete configuration for prompt building and response parsing.
All configuration centralized here - adapters contain no metadata.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import Field, BaseModel

from llm_ensemble.libs.schemas.base_config import BaseConfig


class PromptBuilderAdapterConfig(BaseModel):
    """Nested config for prompt builder adapter instantiation details."""

    prompt_builder_module: str = Field(
        ...,
        description="Full Python module path to prompt builder adapter"
    )
    prompt_builder_class: str = Field(
        ...,
        description="Prompt builder adapter class name in UpperCamelCase"
    )


class PromptSubConfig(BaseModel):
    """Nested config for prompt-specific settings."""

    name_hint: str = Field(
        ...,
        description="Short name hint for this prompt config (used for logging/naming)"
    )
    prompt_name: str = Field(
        ...,
        description="Natural key for Prompt entity (e.g., 'thomas_simple')"
    )
    prompt_template_path: str = Field(
        ...,
        description="Path to prompt template file (relative to prompt templates dir)"
    )
    prompt_builder_adapter: PromptBuilderAdapterConfig = Field(
        ...,
        description="Adapter instantiation configuration for prompt builder"
    )


class ParserAdapterConfig(BaseModel):
    """Nested config for parser adapter instantiation details."""

    parser_module: str = Field(
        ...,
        description="Full Python module path to parser adapter"
    )
    parser_class: str = Field(
        ...,
        description="Parser adapter class name in UpperCamelCase"
    )


class ParserSubConfig(BaseModel):
    """Nested config for parser-specific settings."""

    name_hint: str = Field(
        ...,
        description="Short name hint for this parser config (used for logging/naming)"
    )
    parser_name: str = Field(
        ...,
        description="Natural key for Parser entity (e.g., 'thomas_simple_parser')"
    )
    parser_adapter: ParserAdapterConfig = Field(
        ...,
        description="Adapter instantiation configuration for parser"
    )


class PromptParserConfig(BaseConfig):
    """Complete configuration for prompt building and response parsing.

    All config centralized here - adapters are pure implementation.
    This config includes both prompt and parser identity AND adapter wiring.

    Example YAML:
        name_hint: thomas-simple
        prompt_config:
            name_hint: thomas-simple
            prompt_name: thomas_simple
            prompt_template_path: thomas-simple.jinja
            prompt_builder_adapter:
                prompt_builder_module: llm_ensemble.infer.adapters.prompts.jinja_prompt_builder
                prompt_builder_class: JinjaPromptBuilder
        parser_config:
            name_hint: thomas-simple-parser
            parser_name: thomas_simple_parser
            parser_adapter:
                parser_module: llm_ensemble.infer.adapters.parsers.thomas_simple_parser
                parser_class: ThomasSimpleParser

    Note: name_hint is inherited from BaseConfig and used for run_name generation.
    """

    prompt_config: PromptSubConfig = Field(
        ...,
        description="Prompt configuration including builder adapter"
    )

    parser_config: ParserSubConfig = Field(
        ...,
        description="Parser configuration including parser adapter"
    )

    description: Optional[str] = Field(
        None,
        description="Human-readable description of this prompt-parser combination"
    )

    def get_prompt_builder(self) -> Any:
        """Instantiate and return the prompt builder adapter.

        Dynamically imports the builder module and instantiates the builder class.
        Prompt name and template path come from config.

        Returns:
            Instance of the prompt builder adapter

        Raises:
            ImportError: If the builder module cannot be imported
            AttributeError: If the builder class doesn't exist in the module
        """
        return self._instantiate_adapter(
            self.prompt_config.prompt_builder_adapter.prompt_builder_module,
            self.prompt_config.prompt_builder_adapter.prompt_builder_class,
            prompt_name=self.prompt_config.prompt_name,
            template_path=self.prompt_config.prompt_template_path
        )

    def get_response_parser(self) -> Any:
        """Instantiate and return the response parser adapter.

        Dynamically imports the parser module and instantiates the parser class.
        Parser knows what to look for based on its implementation.

        Returns:
            Instance of the response parser adapter

        Raises:
            ImportError: If the parser module cannot be imported
            AttributeError: If the parser class doesn't exist in the module
        """
        return self._instantiate_adapter(
            self.parser_config.parser_adapter.parser_module,
            self.parser_config.parser_adapter.parser_class
        )
