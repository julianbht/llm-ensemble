"""HuggingFace Inference Endpoints adapter.

This adapter handles all I/O with HuggingFace Inference Endpoints:
- HTTP requests to inference endpoints
- Authentication via HF_TOKEN env var
- Retry logic and error handling
- Response parsing

Follows 12-factor app design: configuration via environment variables.
"""
from __future__ import annotations
import os
import time
from typing import Iterator, Optional
from urllib.parse import urljoin
import json

from llm_ensemble.infer.domain.models import ModelJudgement, ModelConfig
from llm_ensemble.infer.domain.prompts import build_chat_messages
from llm_ensemble.infer.domain.parser import parse_judgement, normalize_confidence


class HuggingFaceInferenceClient:
    """Client for HuggingFace Inference Endpoints.

    Configuration via environment:
    - HF_TOKEN: HuggingFace API token (required)
    - HF_TIMEOUT: Request timeout in seconds (default: 30)
    - HF_MAX_RETRIES: Maximum retry attempts (default: 3)
    """

    def __init__(
        self,
        endpoint_url: str,
        model_config: ModelConfig,
        token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize HuggingFace client.

        Args:
            endpoint_url: Full URL to the inference endpoint
            model_config: Model configuration with default parameters
            token: HF API token (if None, reads from HF_TOKEN env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.endpoint_url = endpoint_url
        self.model_config = model_config
        self.token = token or os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable or pass token parameter."
            )

    def infer(self, example: any, system_prompt: Optional[str] = None) -> ModelJudgement:
        """Run inference on a single example.

        Args:
            example: JudgingExample or compatible object with query/doc fields
            system_prompt: Optional system prompt override

        Returns:
            ModelJudgement with structured output

        Raises:
            RuntimeError: If inference fails after retries
        """
        messages = build_chat_messages(example, system_prompt)

        # Prepare request payload
        payload = {
            "inputs": self._format_messages(messages),
            "parameters": {
                **self.model_config.default_params,
                "return_full_text": False,
            },
        }

        # Execute with retries
        start_time = time.time()
        retries = 0
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(payload)
                latency_ms = (time.time() - start_time) * 1000

                # Extract generated text
                raw_text = self._extract_text(response)

                # Parse into structured judgement
                label, score, rationale, warnings = parse_judgement(raw_text)
                confidence = normalize_confidence(score, label)

                return ModelJudgement(
                    model_id=self.model_config.model_id,
                    provider=self.model_config.provider,
                    version=self.model_config.capabilities.get("version"),
                    query_id=example.query_id,
                    docid=example.docid,
                    label=label,
                    score=score,
                    confidence=confidence,
                    rationale=rationale,
                    raw_text=raw_text,
                    latency_ms=latency_ms,
                    retries=retries,
                    warnings=warnings,
                )

            except Exception as e:
                last_error = e
                retries += 1
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    break

        # All retries exhausted
        raise RuntimeError(
            f"Inference failed after {retries} retries for {example.query_id}/{example.docid}: {last_error}"
        )

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages for the model.

        Different models may expect different formats. This is a simple
        implementation that concatenates system and user messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        # Simple concatenation for now
        # In production, this would use the model's chat template
        formatted = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"[{role}]\n{content}\n")

        return "\n".join(formatted)

    def _post(self, payload: dict) -> dict:
        """Execute HTTP POST to inference endpoint.

        Args:
            payload: Request payload

        Returns:
            Response JSON

        Raises:
            RuntimeError: On HTTP errors
        """
        try:
            import requests
        except ImportError:
            raise RuntimeError(
                "requests library required for HuggingFace adapter. "
                "Install with: pip install requests"
            )

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error: {response.status_code} - {response.text}"
            )

        return response.json()

    def _extract_text(self, response: dict) -> str:
        """Extract generated text from API response.

        Args:
            response: API response JSON

        Returns:
            Generated text

        Raises:
            ValueError: If response format is unexpected
        """
        # HuggingFace Inference API response format varies by task
        # Text generation typically returns: [{"generated_text": "..."}]
        if isinstance(response, list) and len(response) > 0:
            if "generated_text" in response[0]:
                return response[0]["generated_text"]

        # Fallback: try to find any text field
        if isinstance(response, dict):
            for key in ["text", "generated_text", "output"]:
                if key in response:
                    return response[key]

        raise ValueError(f"Could not extract text from response: {response}")


def iter_judgements(
    examples: Iterator[any],
    model_config: ModelConfig,
    endpoint_url: str,
    system_prompt: Optional[str] = None,
) -> Iterator[ModelJudgement]:
    """Generate judgements for a stream of examples.

    Args:
        examples: Iterator of JudgingExample objects
        model_config: Model configuration
        endpoint_url: HF Inference Endpoint URL
        system_prompt: Optional system prompt override

    Yields:
        ModelJudgement objects

    Example:
        >>> examples = iter_examples(Path("data/"))
        >>> judgements = iter_judgements(
        ...     examples,
        ...     model_config,
        ...     "https://api-inference.huggingface.co/models/microsoft/phi-3-mini"
        ... )
        >>> for j in judgements:
        ...     print(j.model_dump_json())
    """
    client = HuggingFaceInferenceClient(
        endpoint_url=endpoint_url,
        model_config=model_config,
    )

    for example in examples:
        yield client.infer(example, system_prompt)
