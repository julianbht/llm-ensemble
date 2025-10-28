"""LLM provider adapters for inference."""

from llm_ensemble.infer.adapters.providers.openrouter_adapter import OpenRouterAdapter
from llm_ensemble.infer.adapters.providers.ollama_adapter import OllamaAdapter
from llm_ensemble.infer.adapters.providers.huggingface_adapter import HuggingFaceAdapter

__all__ = [
    "OpenRouterAdapter",
    "OllamaAdapter",
    "HuggingFaceAdapter",
]
