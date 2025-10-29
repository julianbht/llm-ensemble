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
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser


class OpenRouterAdapter(LLMProvider):
    """OpenRouter implementation of the LLMProvider port.

    Sends inference requests to OpenRouter API and yields (sample, LLMResponse)
    tuples. Uses injected PromptBuilder and ResponseParser ports following
    dependency inversion principles.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> from llm_ensemble.infer.adapters.prompt_builder_factory import get_prompt_builder
        >>> from llm_ensemble.infer.adapters.response_parser_factory import get_response_parser
        >>> config = load_model_config("gpt-oss-20b")
        >>> prompt_config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(prompt_config)
        >>> parser = get_response_parser(prompt_config)
        >>> adapter = OpenRouterAdapter(builder, parser)
        >>> sample_response_pairs = adapter.infer(samples, config)
        >>> for sample, response in sample_response_pairs:
        ...     print(response.llm_score)
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize OpenRouter adapter with injected dependencies.

        Args:
            prompt_builder: PromptBuilder port for building prompts
            response_parser: ResponseParser port for parsing responses
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
        """
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
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

            # Extract response
            raw_response = response.choices[0].message.content

            # Parse the model output using injected parser
            llm_score, parse_warnings = self.response_parser.parse(raw_response)
            warnings.extend(parse_warnings)

            # Build LLMResponse (just the LLM output, no sample or manifest)
            llm_response = LLMResponse(
                llm_score=llm_score,  # RelevanceScore or None if parsing failed
                rationale=None,  # Template doesn't request rationale
                confidence=None,  # Not provided by this template
                raw_response=raw_response,
                latency_ms=latency_ms,
                retries=0,
                cost_estimate_usd=None,  # Could be added later
                warnings=warnings,
            )

            # Yield (sample, response) tuple
            yield (sample, llm_response)
