"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.
"""

from __future__ import annotations
import os
import time
from typing import Iterator, Optional
from openai import OpenAI

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser


class OpenRouterAdapter(LLMProvider):
    """OpenRouter implementation of the LLMProvider port.

    Sends inference requests to OpenRouter API and yields ModelJudgement
    objects. Uses injected PromptBuilder and ResponseParser ports following
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
        >>> judgements = adapter.infer(examples, config)
        >>> for judgement in judgements:
        ...     print(judgement.label)
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
        examples: Iterator[JudgingExample],
        model_config: ModelConfig,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples using OpenRouter API.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            ModelJudgement objects with predictions and metadata

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

        # Process each example
        for example in examples:
            # Build prompt instruction using injected builder
            instruction = self.prompt_builder.build(example)

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
            raw_text = response.choices[0].message.content

            # Parse the model output using injected parser
            label, parse_warnings = self.response_parser.parse(raw_text)
            warnings.extend(parse_warnings)

            # Extract model version if available
            model_version = response.model if hasattr(response, "model") else None

            # Build and yield ModelJudgement
            yield ModelJudgement(
                model_id=model_config.openrouter_model_id,
                provider="openrouter",
                version=model_version,
                query_id=example.query_id,
                docid=example.docid,
                label=label,  # None if parsing failed
                score=float(label) if label is not None else None,  # 0, 1, or 2
                confidence=None,  # Not provided by this template
                rationale=None,  # Template doesn't request rationale
                raw_text=raw_text,
                latency_ms=latency_ms,
                retries=0,
                cost_estimate=None,  # Could be added later
                warnings=warnings,
            )
