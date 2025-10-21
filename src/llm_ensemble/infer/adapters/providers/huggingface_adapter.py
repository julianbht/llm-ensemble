"""HuggingFace adapter for LLM inference.

Handles communication with HuggingFace Inference API and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.

Dependencies (builder, parser, template) are injected via constructor.
"""

from __future__ import annotations
import os
from typing import Iterator, Optional
from jinja2 import Template

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser


class HuggingFaceAdapter(LLMProvider):
    """HuggingFace implementation of the LLMProvider port.

    Sends inference requests to HuggingFace Inference API and yields
    ModelJudgement objects. Receives prompt builder, response parser, and template
    as constructor dependencies (dependency injection pattern).

    Example:
        >>> builder = ThomasPromptBuilder()
        >>> parser = ThomasResponseParser()
        >>> template = load_template("thomas-et-al-prompt.jinja")
        >>> adapter = HuggingFaceAdapter(
        ...     builder=builder,
        ...     parser=parser,
        ...     template=template,
        ...     api_token="..."
        ... )
        >>> judgements = adapter.infer(examples, config)
    """

    def __init__(
        self,
        builder: PromptBuilder,
        parser: ResponseParser,
        template: Template,
        api_token: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize HuggingFace adapter with injected dependencies.

        Args:
            builder: PromptBuilder adapter for building prompts from examples
            parser: ResponseParser adapter for parsing LLM responses
            template: Jinja2 Template for prompt formatting
            api_token: HuggingFace API token (defaults to HF_TOKEN env var)
            timeout: Request timeout in seconds (default: 30)
        """
        # Store injected dependencies
        self.builder = builder
        self.parser = parser
        self.template = template
        
        # API configuration
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.timeout = timeout

        if not self.api_token:
            raise ValueError(
                "HuggingFace API token required. Set HF_TOKEN env var "
                "or pass api_token parameter."
            )

    def infer(
        self,
        examples: Iterator[JudgingExample],
        model_config: ModelConfig,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples using HuggingFace API.

        Uses injected dependencies (builder, parser, template) to process examples.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            ModelJudgement objects with predictions and metadata

        Raises:
            NotImplementedError: HuggingFace adapter not yet implemented
        """
        raise NotImplementedError("HuggingFace adapter not yet implemented")
