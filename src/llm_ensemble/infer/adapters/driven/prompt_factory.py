"""Builder for prompt adapters.

Explicit instantiation of prompt adapters with prompt-specific constructors.
Each prompt adapter defines its own constructor signature and configuration needs.

To add a new prompt:
1. Create adapter class that extends PromptBuilder port
2. Import it here
3. Add explicit instantiation case in create() method
"""

from __future__ import annotations

from llm_ensemble.infer.application.ports.driven.for_building_prompts import ForBuildingPrompts
from llm_ensemble.infer.adapters.driven.prompts.thomas_simple_prompt_builder import ThomasSimplePromptBuilder
from llm_ensemble.infer.adapters.driven.prompts.thomas_advanced_prompt_builder import ThomasAdvancedPromptBuilder


class PromptAdapterFactory:
    """Builder for creating prompt adapter instances."""

    @staticmethod
    def create(prompt_name: str) -> ForBuildingPrompts:
        """Build and return a prompt adapter instance.

        Uses explicit instantiation per prompt to allow prompt-specific
        constructor signatures and configuration.

        Args:
            prompt_name: Name of the prompt (e.g., 'thomas-simple', 'thomas-advanced')

        Returns:
            Instantiated prompt adapter

        Raises:
            ValueError: If prompt not found
        """
        if prompt_name == "thomas-simple":
            return ThomasSimplePromptBuilder()
        elif prompt_name == "thomas-advanced":
            return ThomasAdvancedPromptBuilder()
        else:
            available = ", ".join(sorted(["thomas-simple", "thomas-advanced"]))
            raise ValueError(
                f"Prompt '{prompt_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def list_available() -> list[str]:
        """List all available prompt names.

        Returns:
            Sorted list of prompt names
        """
        return sorted(["thomas-simple", "thomas-advanced"])

    @staticmethod
    def has_prompt(prompt_name: str) -> bool:
        """Check if prompt is available.

        Args:
            prompt_name: Name of the prompt

        Returns:
            True if prompt exists
        """
        return prompt_name in ["thomas-simple", "thomas-advanced"]

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
        # Map prompt names to adapter classes for descriptions
        prompt_classes = {
            "thomas-simple": ThomasSimplePromptBuilder,
            "thomas-advanced": ThomasAdvancedPromptBuilder,
        }

        if prompt_name not in prompt_classes:
            available = ", ".join(sorted(["thomas-simple", "thomas-advanced"]))
            raise ValueError(
                f"Prompt '{prompt_name}' not found. "
                f"Available: {available}"
            )

        adapter_class = prompt_classes[prompt_name]
        return adapter_class.__doc__.strip().split('\n')[0] if adapter_class.__doc__ else prompt_name
