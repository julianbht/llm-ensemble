"""Prompt configuration schema.

Defines the Pydantic schema for prompt template configurations.
"""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Configuration for a prompt template.

    Specifies which variables to pass to the template and any metadata
    about the prompt's purpose and expected outputs.
    """

    name: str = Field(..., description="Prompt identifier (matches filename without .jinja)")
    template_file: str = Field(..., description="Template filename (e.g., 'thomas-et-al-prompt.jinja')")
    description: Optional[str] = Field(None, description="Human-readable description of the prompt")

    # Variables to pass to the Jinja2 template
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to pass when rendering the template (e.g., role=True, aspects=False)"
    )

    # Metadata about expected outputs
    expected_output_format: Optional[str] = Field(
        None,
        description="Expected format of model response (e.g., 'json', 'text')"
    )
    response_parser: Optional[str] = Field(
        None,
        description="Name of parser function to use (e.g., 'parse_thomas_response')"
    )
