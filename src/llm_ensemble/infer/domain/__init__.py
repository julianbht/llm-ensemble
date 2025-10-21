"""Domain layer for the infer CLI.

Contains pure business logic with no infrastructure dependencies.
The domain service orchestrates the inference pipeline by coordinating
port abstractions, maintaining complete independence from concrete
implementations.
"""

from llm_ensemble.infer.domain.inference_service import InferenceService

__all__ = ["InferenceService"]
