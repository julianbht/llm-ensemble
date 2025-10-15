"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to ModelJudgement domain objects.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional, Any
from openai import OpenAI
from jinja2 import Template

from llm_ensemble.infer.domain.prompt_builder import build_instruction_from_judging_example
from llm_ensemble.infer.domain.response_parser import parse_thomas_response


def load_thomas_template() -> Template:
    """Load the thomas-et-al prompt template from disk.

    Returns:
        Jinja2 Template object ready for rendering

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    # Navigate from this file to the template location
    template_path = (
        Path(__file__).parents[1]
        / "domain"
        / "prompts"
        / "thomas-et-al-prompt.jinja"
    )

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        return Template(f.read())


def send_inference_request(
    example: dict,
    model_id: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 30,
) -> dict[str, Any]:
    """Send an inference request to OpenRouter and return a ModelJudgement dict.

    Args:
        example: JudgingExample dict with query_text, doc, query_id, docid
        model_id: OpenRouter model ID (e.g., "qwen/qwen-0.5b-chat")
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

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

    # Load template and build instruction
    template = load_thomas_template()
    instruction = build_instruction_from_judging_example(
        template=template,
        example=example,
        role=True,
        aspects=False,  # Start with simple O-only format
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

    # Parse the model output
    label, parse_warnings = parse_thomas_response(raw_text)
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
