"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to LLMResponse objects. Implements the LLMProvider port.

This is a PURE API client - it accepts pre-built prompts and returns raw responses.
It does NOT build prompts (that's PromptBuilder's job) or parse responses (that's
ResponseParser's job). The InferenceService orchestrates all port interactions.
"""

from __future__ import annotations
import os
import time
from typing import Iterator, Optional
from openai import OpenAI

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider


class OpenRouterAdapter(LLMProvider):
    """OpenRouter implementation of the LLMProvider port.

    Pure API client that sends pre-built prompts to OpenRouter and returns raw responses.
    Does NOT build prompts or parse responses - that's orchestrated by InferenceService.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> config = load_model_config("gpt-oss-20b")
        >>> adapter = OpenRouterAdapter(api_key="...")
        >>> sample_prompt_pairs = [(sample, "pre-built prompt"), ...]
        >>> sample_response_pairs = adapter.infer(sample_prompt_pairs, config)
        >>> for sample, response in sample_response_pairs:
        ...     print(response.raw_response)  # Unparsed text
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize OpenRouter adapter.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

    def infer(
        self,
        sample_prompt_pairs: Iterator[tuple[JudgingSample, str]],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Run inference on pre-built prompts using OpenRouter API.

        Args:
            sample_prompt_pairs: Iterator of (JudgingSample, prompt_string) tuples
                                where prompts have been pre-built by PromptBuilder
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

        # Process each (sample, prompt) pair
        for sample, prompt in sample_prompt_pairs:
            # Track timing
            warnings: list = []  # Will be populated with ProviderWarning objects if needed
            start_time = time.time()

            # Send request with all configured parameters
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
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
