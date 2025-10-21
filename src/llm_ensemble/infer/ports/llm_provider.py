"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig


class LLMProvider(ABC):
    """Abstract base class for LLM inference providers.

    All provider adapters (OpenRouter, Ollama, HuggingFace) must inherit
    from this class and implement the infer() method.

    Example:
        >>> class OpenRouterAdapter(LLMProvider):
        ...     def infer(self, examples, model_config, ...):
        ...         # Implementation
        ...         yield judgement
    """

    @abstractmethod
    def infer(
        self,
        examples: Iterator[JudgingExample],
        model_config: ModelConfig,
        prompt_template_name: str,
        prompts_dir: Optional[Path] = None,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples and yield judgements.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings
            prompt_template_name: Name of the prompt template to use
            prompts_dir: Directory containing prompt templates (defaults to configs/prompts)

        Yields:
            ModelJudgement objects with predictions and metadata

        Raises:
            ValueError: If configuration is invalid
            Exception: If provider API fails
        """
        pass
