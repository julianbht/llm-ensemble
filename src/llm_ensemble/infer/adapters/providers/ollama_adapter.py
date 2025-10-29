"""Ollama adapter for LLM inference.

Handles communication with local Ollama server and converts responses
to LLMResponse objects. Implements the LLMProvider port.

This is a PURE API client - it accepts pre-built prompts and returns raw responses.
It does NOT build prompts (that's PromptBuilder's job) or parse responses (that's
ResponseParser's job). The InferenceService orchestrates all port interactions.
"""

from __future__ import annotations
from typing import Iterator

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider


class OllamaAdapter(LLMProvider):
    """Ollama implementation of the LLMProvider port.

    Pure API client that sends pre-built prompts to Ollama and returns raw responses.
    Does NOT build prompts or parse responses - that's orchestrated by InferenceService.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> config = load_model_config("tinyllama")
        >>> adapter = OllamaAdapter(base_url="http://localhost:11434")
        >>> sample_prompt_pairs = [(sample, "pre-built prompt"), ...]
        >>> sample_response_pairs = adapter.infer(sample_prompt_pairs, config)
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
        sample_prompt_pairs: Iterator[tuple[JudgingSample, str]],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Run inference on pre-built prompts using Ollama server.

        Args:
            sample_prompt_pairs: Iterator of (JudgingSample, prompt_string) tuples
                                where prompts have been pre-built by PromptBuilder
            model_config: Model configuration with provider and settings

        Yields:
            Tuples of (JudgingSample, LLMResponse) for each inference

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
