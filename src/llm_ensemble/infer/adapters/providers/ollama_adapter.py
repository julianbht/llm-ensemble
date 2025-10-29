"""Ollama adapter for LLM inference.

Handles communication with local Ollama server and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.
"""

from __future__ import annotations
from typing import Iterator

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder


class OllamaAdapter(LLMProvider):
    """Ollama implementation of the LLMProvider port.

    Sends inference requests to local Ollama server and yields (sample, LLMResponse)
    tuples with RAW responses only. Uses injected PromptBuilder for prompt construction.

    Providers do NOT parse responses - they return raw text + metadata.
    The InferenceService coordinates parsing via ResponseParser.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> from llm_ensemble.infer.adapters.prompt_builder_factory import get_prompt_builder
        >>> config = load_model_config("tinyllama")
        >>> prompt_config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(prompt_config)
        >>> adapter = OllamaAdapter(builder)
        >>> sample_response_pairs = adapter.infer(samples, config)
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        """Initialize Ollama adapter with injected dependencies.

        Args:
            prompt_builder: PromptBuilder port for building prompts
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
        """
        self.prompt_builder = prompt_builder
        self.base_url = base_url
        self.timeout = timeout

    def infer(
        self,
        samples: Iterator[JudgingSample],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Run inference on samples using Ollama server.

        Args:
            samples: Iterator of JudgingSample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            Tuples of (JudgingSample, LLMResponse) for each inference

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
