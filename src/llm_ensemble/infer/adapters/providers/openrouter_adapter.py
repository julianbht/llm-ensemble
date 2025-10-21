"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Iterator, Optional
from openai import OpenAI

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider
from llm_ensemble.infer.config_loaders import load_prompt_config, load_prompt_template
from llm_ensemble.infer.prompt_builders import load_builder
from llm_ensemble.infer.response_parsers import load_parser


class OpenRouterAdapter(LLMProvider):
    """OpenRouter implementation of the LLMProvider port.

    Sends inference requests to OpenRouter API and yields ModelJudgement
    objects. Uses prompt builders and response parsers for flexible
    prompt formatting and output parsing.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> config = load_model_config("gpt-oss-20b")
        >>> adapter = OpenRouterAdapter()
        >>> judgements = adapter.infer(examples, config, "thomas-et-al-prompt")
        >>> for judgement in judgements:
        ...     print(judgement.label)
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
        examples: Iterator[JudgingExample],
        model_config: ModelConfig,
        prompt_template_name: str,
        prompts_dir: Optional[Path] = None,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples using OpenRouter API.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings
            prompt_template_name: Name of the prompt template to use
            prompts_dir: Directory containing prompt templates (defaults to configs/prompts)

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

        # Load prompt config and template
        prompt_config = load_prompt_config(prompt_template_name, prompts_dir)
        template = load_prompt_template(prompt_config.prompt_template, prompts_dir)

        # Load builder and parser
        builder = load_builder(prompt_config.prompt_builder)
        parser = load_parser(prompt_config.response_parser)

        # Extract model parameters
        temperature = model_config.default_params.get("temperature", 0.0)
        max_tokens = model_config.default_params.get("max_tokens", 256)

        # Initialize OpenAI client configured for OpenRouter
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=self.timeout,
        )

        # Process each example
        for example in examples:
            # Convert example to dict for builder
            example_dict = example.model_dump()

            # Build prompt instruction
            instruction = builder.build(template, example_dict)

            # Track timing
            warnings = []
            start_time = time.time()

            # Send request
            response = client.chat.completions.create(
                model=model_config.openrouter_model_id,
                messages=[{"role": "user", "content": instruction}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            raw_text = response.choices[0].message.content

            # Parse the model output
            label, parse_warnings = parser.parse(raw_text)
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
