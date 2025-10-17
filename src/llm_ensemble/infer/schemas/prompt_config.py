"""Prompt configuration schema.

Defines the Pydantic schema for prompt template configurations.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Configuration for a prompt template.

    Specifies the template file, builder module, and parser module to use.
    """

    name: str = Field(..., description="Prompt identifier")
    description: Optional[str] = Field(None, description="Human-readable description of the prompt")
    prompt_template: str = Field(..., description="Template filename (without .jinja extension)")
    prompt_builder: str = Field(..., description="Name of builder module in prompt_builders/")
    response_parser: str = Field(..., description="Name of parser module in response_parsers/")
