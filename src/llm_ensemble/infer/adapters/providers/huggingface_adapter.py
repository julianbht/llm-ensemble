"""HuggingFace adapter for LLM inference.

Handles communication with HuggingFace Inference API and converts responses
to ModelJudgement domain objects. Implements the LLMProvider port.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Iterator, Optional

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig
from llm_ensemble.infer.ports import LLMProvider


class HuggingFaceAdapter(LLMProvider):
    """HuggingFace implementation of the LLMProvider port.

    Sends inference requests to HuggingFace Inference API and yields
    ModelJudgement objects.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> config = load_model_config("phi3-mini")
        >>> adapter = HuggingFaceAdapter()
        >>> judgements = adapter.infer(examples, config, "thomas-et-al-prompt")
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize HuggingFace adapter.

        Args:
            api_token: HuggingFace API token (defaults to HF_TOKEN env var)
            timeout: Request timeout in seconds (default: 30)
        """
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
        prompt_template_name: str,
        prompts_dir: Optional[Path] = None,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples using HuggingFace API.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings
            prompt_template_name: Name of the prompt template to use
            prompts_dir: Directory containing prompt templates

        Yields:
            ModelJudgement objects with predictions and metadata

        Raises:
            NotImplementedError: HuggingFace adapter not yet implemented
        """
        raise NotImplementedError("HuggingFace adapter not yet implemented")
