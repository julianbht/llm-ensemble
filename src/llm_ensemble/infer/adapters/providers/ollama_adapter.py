"""Ollama adapter for LLM inference.

Handles communication with local Ollama server and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.

Dependencies (builder, parser, template) are injected via constructor.
"""

from __future__ import annotations
from typing import Iterator
from jinja2 import Template

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser


class OllamaAdapter(LLMProvider):
    """Ollama implementation of the LLMProvider port.

    Sends inference requests to local Ollama server and yields ModelJudgement
    objects. Receives prompt builder, response parser, and template as
    constructor dependencies (dependency injection pattern).

    Example:
        >>> builder = ThomasPromptBuilder()
        >>> parser = ThomasResponseParser()
        >>> template = load_template("thomas-et-al-prompt.jinja")
        >>> adapter = OllamaAdapter(
        ...     builder=builder,
        ...     parser=parser,
        ...     template=template,
        ... )
        >>> judgements = adapter.infer(examples, config)
    """

    def __init__(
        self,
        builder: PromptBuilder,
        parser: ResponseParser,
        template: Template,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        """Initialize Ollama adapter with injected dependencies.

        Args:
            builder: PromptBuilder adapter for building prompts from examples
            parser: ResponseParser adapter for parsing LLM responses
            template: Jinja2 Template for prompt formatting
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
        """
        # Store injected dependencies
        self.builder = builder
        self.parser = parser
        self.template = template
        
        # API configuration
        self.base_url = base_url
        self.timeout = timeout

    def infer(
        self,
        examples: Iterator[JudgingExample],
        model_config: ModelConfig,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples using Ollama server.

        Uses injected dependencies (builder, parser, template) to process examples.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            ModelJudgement objects with predictions and metadata

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
