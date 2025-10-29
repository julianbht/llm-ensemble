"""HuggingFace adapter for LLM inference.

Handles communication with HuggingFace Inference API and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.
"""

from __future__ import annotations
import os
from typing import Iterator, Optional

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser


class HuggingFaceAdapter(LLMProvider):
    """HuggingFace implementation of the LLMProvider port.

    Sends inference requests to HuggingFace Inference API and yields
    (sample, LLMResponse) tuples. Uses injected PromptBuilder and ResponseParser
    ports following dependency inversion principles.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> from llm_ensemble.infer.adapters.prompt_builder_factory import get_prompt_builder
        >>> from llm_ensemble.infer.adapters.response_parser_factory import get_response_parser
        >>> config = load_model_config("phi3-mini")
        >>> prompt_config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(prompt_config)
        >>> parser = get_response_parser(prompt_config)
        >>> adapter = HuggingFaceAdapter(builder, parser)
        >>> sample_response_pairs = adapter.infer(samples, config)
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        api_token: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize HuggingFace adapter with injected dependencies.

        Args:
            prompt_builder: PromptBuilder port for building prompts
            response_parser: ResponseParser port for parsing responses
            api_token: HuggingFace API token (defaults to HF_TOKEN env var)
            timeout: Request timeout in seconds (default: 30)
        """
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.timeout = timeout

        if not self.api_token:
            raise ValueError(
                "HuggingFace API token required. Set HF_TOKEN env var "
                "or pass api_token parameter."
            )

    def infer(
        self,
        samples: Iterator[JudgingSample],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Run inference on samples using HuggingFace API.

        Args:
            samples: Iterator of JudgingSample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            Tuples of (JudgingSample, LLMResponse) for each inference

        Raises:
            NotImplementedError: HuggingFace adapter not yet implemented
        """
        raise NotImplementedError("HuggingFace adapter not yet implemented")
