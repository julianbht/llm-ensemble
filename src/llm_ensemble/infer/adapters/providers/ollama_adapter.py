"""Ollama adapter for LLM inference.

Handles communication with local Ollama server and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider


class OllamaAdapter(LLMProvider):
    """Ollama implementation of the LLMProvider port.

    Sends inference requests to local Ollama server and yields ModelJudgement
    objects.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> config = load_model_config("tinyllama")
        >>> adapter = OllamaAdapter()
        >>> judgements = adapter.infer(examples, config, "thomas-et-al-prompt")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        """Initialize Ollama adapter.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
        """
        self.base_url = base_url
        self.timeout = timeout

    def infer(
        self,
        examples: Iterator[JudgingExample],
        model_config: ModelConfig,
        prompt_template_name: str,
        prompts_dir: Optional[Path] = None,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples using Ollama server.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings
            prompt_template_name: Name of the prompt template to use
            prompts_dir: Directory containing prompt templates

        Yields:
            ModelJudgement objects with predictions and metadata

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
