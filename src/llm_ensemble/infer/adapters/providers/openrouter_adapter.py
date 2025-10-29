"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to LLMResponse objects. Implements the LLMProvider port.
"""

from __future__ import annotations
import os
import time
from typing import Iterator, Optional
from openai import OpenAI

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder


class OpenRouterAdapter(LLMProvider):
    """OpenRouter implementation of the LLMProvider port.

    Sends inference requests to OpenRouter API and yields (sample, LLMResponse)
    tuples with RAW responses only. Uses injected PromptBuilder for prompt construction.

    Providers do NOT parse responses - they return raw text + metadata.
    The InferenceService coordinates parsing via ResponseParser.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> from llm_ensemble.infer.adapters.prompt_builder_factory import get_prompt_builder
        >>> config = load_model_config("gpt-oss-20b")
        >>> prompt_config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(prompt_config)
        >>> adapter = OpenRouterAdapter(builder)
        >>> sample_response_pairs = adapter.infer(samples, config)
        >>> for sample, response in sample_response_pairs:
        ...     print(response.raw_response)  # Unparsed text
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize OpenRouter adapter with injected dependencies.

        Args:
            prompt_builder: PromptBuilder port for building prompts
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
        """
        self.prompt_builder = prompt_builder
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

    def infer(
        self,
        samples: Iterator[JudgingSample],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Run inference on samples using OpenRouter API.

        Args:
            samples: Iterator of JudgingSample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            Tuples of (JudgingSample, LLMResponse) for each inference

        Raises:
            ValueError: If openrouter_model_id is not configured
            Exception: If API request fails
        """
        if not model_config.openrouter_model_id:
            raise ValueError(
                f"Model {model_config.model_id} is configured for OpenRouter "
                f"but missing openrouter_model_id field"
            )

        # Build API parameters from explicit config fields
        api_params = {
            "model": model_config.openrouter_model_id,
        }

        # Add explicit parameters if set
        if model_config.temperature is not None:
            api_params["temperature"] = model_config.temperature
        if model_config.max_tokens is not None:
            api_params["max_tokens"] = model_config.max_tokens
        if model_config.top_p is not None:
            api_params["top_p"] = model_config.top_p
        if model_config.frequency_penalty is not None:
            api_params["frequency_penalty"] = model_config.frequency_penalty
        if model_config.presence_penalty is not None:
            api_params["presence_penalty"] = model_config.presence_penalty
        if model_config.seed is not None:
            api_params["seed"] = model_config.seed
        if model_config.stop is not None:
            api_params["stop"] = model_config.stop
        if model_config.response_format is not None:
            api_params["response_format"] = model_config.response_format

        # Add additional parameters (advanced/provider-specific)
        api_params.update(model_config.additional_params)

        # Initialize OpenAI client configured for OpenRouter
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=self.timeout,
        )

        # Process each sample
        for sample in samples:
            # Build prompt instruction using injected builder
            instruction = self.prompt_builder.build(sample)

            # Track timing
            warnings = []
            start_time = time.time()

            # Send request with all configured parameters
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": instruction}],
                **api_params
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response text
            raw_response = response.choices[0].message.content

            # Build LLMResponse with RAW output only (no parsing)
            # The InferenceService will handle parsing via ResponseParser
            llm_response = LLMResponse(
                raw_response=raw_response,
                latency_ms=latency_ms,
                retries=0,
                cost_estimate_usd=None,  # Could be added later
                warnings=warnings,
            )

            # Yield (sample, raw_response) tuple
            yield (sample, llm_response)
