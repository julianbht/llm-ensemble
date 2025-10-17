"""Providers feature for LLM inference across different providers."""

from llm_ensemble.infer.providers.openrouter import send_inference_request

__all__ = [
    "send_inference_request",
]
