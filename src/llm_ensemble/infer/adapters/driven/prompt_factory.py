"""Builder for prompt adapters.

Explicit instantiation of prompt adapters with prompt-specific constructors.
Each prompt adapter defines its own constructor signature and configuration needs.

To add a new prompt:
1. Create adapter class that extends PromptBuilder port
2. Import it here
3. Add explicit instantiation case in create() method
"""

from __future__ import annotations

from llm_ensemble.infer.adapters.driven.prompts.thomas_advanced_trec_prompt_builder import (
    ThomasAdvancedTrecPromptBuilder,
)
from llm_ensemble.infer.adapters.driven.prompts.thomas_simple_prompt_builder import (
    ThomasSimplePromptBuilder,
)
from llm_ensemble.infer.application.ports.driven.for_building_prompts import (
    ForBuildingPrompts,
)


AVAILABLE_PROMPTS = ["thomas-simple", "thomas-advanced-trec-v2"]


class PromptAdapterFactory:
    """Builder for creating prompt adapter instances."""

    @staticmethod
    def create(prompt_name: str) -> ForBuildingPrompts:
        """Build and return a prompt adapter instance.

        Uses explicit instantiation per prompt to allow prompt-specific
        constructor signatures and configuration.

        Args:
            prompt_name: Name of the prompt (e.g., 'thomas-simple', 'thomas-advanced-trec')

        Returns:
            Instantiated prompt adapter

        Raises:
            ValueError: If prompt not found
        """
        if prompt_name == "thomas-simple":
            return ThomasSimplePromptBuilder()
        elif prompt_name == "thomas-advanced-trec-v2":
            return ThomasAdvancedTrecPromptBuilder()
        else:
            available = ", ".join(sorted(AVAILABLE_PROMPTS))
            raise ValueError(
                f"Prompt '{prompt_name}' not found. " f"Available: {available}"
            )

    @staticmethod
    def list_available() -> list[str]:
        """List all available prompt names.

        Returns:
            Sorted list of prompt names
        """
        return sorted(AVAILABLE_PROMPTS)

    @staticmethod
    def has_prompt(prompt_name: str) -> bool:
        """Check if prompt is available.

        Args:
            prompt_name: Name of the prompt

        Returns:
            True if prompt exists
        """
        return prompt_name in AVAILABLE_PROMPTS
