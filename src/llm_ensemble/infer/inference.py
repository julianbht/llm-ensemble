"""Main inference orchestration.

Routes inference requests to the appropriate provider adapter based on model config.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional

from llm_ensemble.ingest.domain.models import JudgingExample
from llm_ensemble.infer.models import ModelJudgement
from llm_ensemble.infer.config.models import ModelConfig
from llm_ensemble.infer.providers.openrouter import send_inference_request


def iter_judgements(
    examples: Iterator[JudgingExample],
    model_config: ModelConfig,
    prompts_dir: Optional[Path] = None,
    prompt_template_name: str = "thomas-et-al-prompt",
) -> Iterator[ModelJudgement]:
    """Run inference over examples using the appropriate provider adapter.

    Args:
        examples: Iterator of JudgingExample objects
        model_config: Model configuration with provider and settings
        prompts_dir: Directory containing prompt templates (defaults to configs/prompts)
        prompt_template_name: Name of the prompt template to use (defaults to "thomas-et-al-prompt")

    Yields:
        ModelJudgement objects for each example

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> from llm_ensemble.infer.config import load_model_config
        >>> config = load_model_config("gpt-oss-20b")
        >>> examples = [example1, example2]
        >>> for judgement in iter_judgements(iter(examples), config):
        ...     print(judgement.label)
    """
    if model_config.provider == "openrouter":
        yield from _iter_openrouter_judgements(examples, model_config, prompts_dir, prompt_template_name)
    elif model_config.provider == "hf":
        raise NotImplementedError("HuggingFace adapter not yet implemented")
    elif model_config.provider == "ollama":
        raise NotImplementedError("Ollama adapter not yet implemented")
    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")


def _iter_openrouter_judgements(
    examples: Iterator[JudgingExample],
    model_config: ModelConfig,
    prompts_dir: Optional[Path] = None,
    prompt_template_name: str = "thomas-et-al-prompt",
) -> Iterator[ModelJudgement]:
    """Run inference using OpenRouter adapter.

    Args:
        examples: Iterator of JudgingExample objects
        model_config: Model configuration
        prompts_dir: Directory containing prompt templates (defaults to configs/prompts)
        prompt_template_name: Name of the prompt template to use (defaults to "thomas-et-al-prompt")

    Yields:
        ModelJudgement objects

    Raises:
        ValueError: If openrouter_model_id is not configured
    """
    if not model_config.openrouter_model_id:
        raise ValueError(
            f"Model {model_config.model_id} is configured for OpenRouter "
            f"but missing openrouter_model_id field"
        )

    # Extract default params
    temperature = model_config.default_params.get("temperature", 0.0)
    max_tokens = model_config.default_params.get("max_tokens", 256)

    for example in examples:
        # Convert JudgingExample to dict for the adapter
        example_dict = example.model_dump()

        # Send inference request
        judgement_dict = send_inference_request(
            example=example_dict,
            model_id=model_config.openrouter_model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_template_name=prompt_template_name,
            prompts_dir=prompts_dir,
        )

        # Convert dict to ModelJudgement object
        yield ModelJudgement(**judgement_dict)
