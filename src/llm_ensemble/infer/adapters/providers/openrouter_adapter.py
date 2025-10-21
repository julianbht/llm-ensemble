"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.

Dependencies (builder, parser, template) are injected via constructor,
following dependency injection principle.
"""

from __future__ import annotations
import os
import time
from typing import Iterator, Optional
from openai import OpenAI
from jinja2 import Template

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser


class OpenRouterAdapter(LLMProvider):
    """OpenRouter implementation of the LLMProvider port.

    Sends inference requests to OpenRouter API and yields ModelJudgement
    objects. Receives prompt builder, response parser, and template as
    constructor dependencies (dependency injection pattern).

    Example:
        >>> builder = ThomasPromptBuilder()
        >>> parser = ThomasResponseParser()
        >>> template = load_template("thomas-et-al-prompt.jinja")
        >>> adapter = OpenRouterAdapter(
        ...     builder=builder,
        ...     parser=parser,
        ...     template=template,
        ...     api_key="..."
        ... )
        >>> judgements = adapter.infer(examples, config)
        >>> for judgement in judgements:
        ...     print(judgement.label)
    """

    def __init__(
        self,
        builder: PromptBuilder,
        parser: ResponseParser,
        template: Template,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize OpenRouter adapter with injected dependencies.

        Args:
            builder: PromptBuilder adapter for building prompts from examples
            parser: ResponseParser adapter for parsing LLM responses
            template: Jinja2 Template for prompt formatting
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
        """
        # Store injected dependencies
        self.builder = builder
        self.parser = parser
        self.template = template
        
        # API configuration
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

        Uses injected dependencies (builder, parser, template) to process examples.
        No config loading needed - all dependencies already injected.

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

            # Build prompt instruction using injected builder and template
            instruction = self.builder.build(self.template, example_dict)

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

            # Parse the model output using injected parser
            label, parse_warnings = self.parser.parse(raw_text)
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
