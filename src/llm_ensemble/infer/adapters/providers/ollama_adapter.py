"""Ollama adapter for LLM inference.

Handles communication with local Ollama server and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.
"""

from __future__ import annotations
from typing import Iterator

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser


class OllamaAdapter(LLMProvider):
    """Ollama implementation of the LLMProvider port.

    Sends inference requests to local Ollama server and yields (sample, LLMResponse)
    tuples. Uses injected PromptBuilder and ResponseParser ports following
    dependency inversion principles.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> from llm_ensemble.infer.adapters.prompt_builder_factory import get_prompt_builder
        >>> from llm_ensemble.infer.adapters.response_parser_factory import get_response_parser
        >>> config = load_model_config("tinyllama")
        >>> prompt_config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(prompt_config)
        >>> parser = get_response_parser(prompt_config)
        >>> adapter = OllamaAdapter(builder, parser)
        >>> sample_response_pairs = adapter.infer(samples, config)
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        """Initialize Ollama adapter with injected dependencies.

        Args:
            prompt_builder: PromptBuilder port for building prompts
            response_parser: ResponseParser port for parsing responses
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
        """
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
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
