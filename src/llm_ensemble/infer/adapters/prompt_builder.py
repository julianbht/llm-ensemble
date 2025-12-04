"""Builder for prompt adapters.

Simple, explicit mapping of prompt names to adapter classes.
No decorators, no hidden registration - just a clear dictionary.

To add a new prompt:
1. Create adapter class that extends PromptBuilder port
2. Import it here
3. Add to PROMPTS dict
"""

from __future__ import annotations
from typing import Dict, Type

from llm_ensemble.infer.ports import PromptBuilderPort
from llm_ensemble.infer.adapters.prompts.thomas_simple_prompt_builder import (
    ThomasSimplePromptBuilder,
)


# Explicit mapping of prompt names to adapter classes
# Each adapter owns its template as a class constant
PROMPTS: Dict[str, Type[PromptBuilderPort]] = {
    "thomas-simple": ThomasSimplePromptBuilder,
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

        adapter_class = PROMPTS[prompt_name]
        return adapter_class()

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
            Description string from adapter's docstring

        Raises:
            ValueError: If prompt not found
        """
        if prompt_name not in PROMPTS:
            available = ", ".join(sorted(PROMPTS.keys()))
            raise ValueError(
                f"Prompt '{prompt_name}' not found. "
                f"Available: {available}"
            )

        adapter_class = PROMPTS[prompt_name]
        return adapter_class.__doc__.strip().split('\n')[0] if adapter_class.__doc__ else prompt_name
