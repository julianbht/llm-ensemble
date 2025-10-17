"""OpenRouter provider for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to ModelJudgement domain objects.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional, Any
from openai import OpenAI

from llm_ensemble.infer.prompts.builder import build_instruction_from_judging_example
from llm_ensemble.infer.parsers.thomas import load_parser
from llm_ensemble.infer.prompts.templates import load_prompt_template
from llm_ensemble.infer.config.prompts_loader import load_prompt_config


def send_inference_request(
    example: dict,
    model_id: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 30,
    prompt_template_name: str = "thomas-et-al-prompt",
    prompts_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Send an inference request to OpenRouter and return a ModelJudgement dict.

    Args:
        example: JudgingExample dict with query_text, doc, query_id, docid
        model_id: OpenRouter model ID (e.g., "qwen/qwen-0.5b-chat")
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        prompt_template_name: Name of the prompt template to use (defaults to "thomas-et-al-prompt")
        prompts_dir: Directory containing prompt templates (defaults to configs/prompts)

    Returns:
        Dict matching ModelJudgement schema with all required fields

    Raises:
        ValueError: If API key is not provided and not in environment
        Exception: If the API request fails

    Example:
        >>> example = {
        ...     "query_text": "python tutorial",
        ...     "doc": "Learn Python...",
        ...     "query_id": "q1",
        ...     "docid": "d1"
        ... }
        >>> result = send_inference_request(example, "qwen/qwen-0.5b-chat")
        >>> result["label"]  # 0, 1, or 2
        2
    """
    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
            "or pass api_key parameter."
        )

    # Load prompt config and template
    prompt_config = load_prompt_config(prompt_template_name, prompts_dir)
    template = load_prompt_template(prompt_config.template_file, prompts_dir)

    # Build instruction using variables from config
    instruction = build_instruction_from_judging_example(
        template=template,
        example=example,
        **prompt_config.variables  # Unpack variables from config
    )

    # Initialize OpenAI client configured for OpenRouter
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout,
    )

    # Track timing
    warnings = []
    start_time = time.time()

    # Send request
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": instruction}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    latency_ms = (time.time() - start_time) * 1000

    # Extract response
    raw_text = response.choices[0].message.content

    # Load parser dynamically from config
    parser = load_parser(prompt_config.response_parser)

    # Parse the model output
    label, parse_warnings = parser(raw_text)
    warnings.extend(parse_warnings)

    # Extract model version if available
    model_version = response.model if hasattr(response, "model") else None

    # Build ModelJudgement dict
    return {
        "model_id": model_id,
        "provider": "openrouter",
        "version": model_version,
        "query_id": example["query_id"],
        "docid": example["docid"],
        "label": label,  # None if parsing failed
        "score": float(label) if label is not None else None,  # 0, 1, or 2 (or None)
        "confidence": None,  # Not provided by this template
        "rationale": None,  # Template doesn't request rationale
        "raw_text": raw_text,
        "latency_ms": latency_ms,
        "retries": 0,
        "cost_estimate": None,  # Could be added later
        "warnings": warnings,
    }
