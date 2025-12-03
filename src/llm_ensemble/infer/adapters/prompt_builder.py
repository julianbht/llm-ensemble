"""Builder for prompt adapters.

Simple, explicit mapping of prompt names to adapter classes.
No decorators, no hidden registration - just a clear dictionary.

To add a new prompt:
1. Create adapter class that extends PromptBuilder port
2. Import it here
3. Add to PROMPTS dict
"""

from __future__ import annotations
from typing import Dict, Type, NamedTuple

from llm_ensemble.infer.ports import PromptBuilderPort
from llm_ensemble.infer.adapters.prompts.jinja_prompt_builder import (
    ThomasSimplePromptBuilder,
)


class PromptConfig(NamedTuple):
    """Configuration for a prompt adapter."""
    adapter_class: Type[PromptBuilderPort]
    template_path: str
    description: str


# Explicit mapping of prompt names to adapter configurations
PROMPTS: Dict[str, PromptConfig] = {
    "thomas-simple": PromptConfig(
        adapter_class=ThomasSimplePromptBuilder,
        template_path="thomas-simple.jinja",
        description="Thomas et al. simple binary relevance prompt",
    ),
}


class PromptAdapterBuilder:
    """Builder for creating prompt adapter instances."""

    @staticmethod
    def build(prompt_name: str) -> PromptBuilderPort:
        """Build and return a prompt adapter instance.

        Args:
            prompt_name: Name of the prompt (e.g., 'thomas-simple')

        Returns:
            Instantiated prompt adapter

        Raises:
            ValueError: If prompt not found
        """
        if prompt_name not in PROMPTS:
            available = ", ".join(sorted(PROMPTS.keys()))
            raise ValueError(
                f"Prompt '{prompt_name}' not found. "
                f"Available: {available}"
            )

        config = PROMPTS[prompt_name]
        return config.adapter_class(
            prompt_name=prompt_name,
            template_path=config.template_path
        )

    @staticmethod
    def list_available() -> list[str]:
        """List all available prompt names.

        Returns:
            Sorted list of prompt names
        """
        return sorted(PROMPTS.keys())

    @staticmethod
    def has_prompt(prompt_name: str) -> bool:
        """Check if prompt is available.

        Args:
            prompt_name: Name of the prompt

        Returns:
            True if prompt exists
        """
        return prompt_name in PROMPTS

    @staticmethod
    def get_description(prompt_name: str) -> str:
        """Get description for a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            Description string

        Raises:
            ValueError: If prompt not found
        """
        if prompt_name not in PROMPTS:
            available = ", ".join(sorted(PROMPTS.keys()))
            raise ValueError(
                f"Prompt '{prompt_name}' not found. "
                f"Available: {available}"
            )

        return PROMPTS[prompt_name].description
