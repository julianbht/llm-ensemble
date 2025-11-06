"""Domain entities for INFER CLI.

These are not configs - they are domain entities that capture experimental parameters.
Configs are just YAML plumbing that points to adapter classes.
These entities capture what actually affects LLM behavior and experimental outcomes.
"""

from __future__ import annotations
from uuid import UUID
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from llm_ensemble.libs.db.uuid_helpers import (
    compute_provider_uuid,
    compute_prompt_template_uuid,
    compute_model_spec_uuid,
)


class Provider(BaseModel):
    """Provider domain entity - LLM service provider.

    Represents which service is used (openrouter, ollama, hf).
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from provider name"
    )

    name: str = Field(
        ...,
        description="Provider name (e.g., 'openrouter', 'ollama', 'hf')"
    )

    @classmethod
    def create(cls, name: str) -> "Provider":
        """Create a Provider with deterministic UUID.

        Args:
            name: Provider name (e.g., "openrouter", "ollama", "hf")

        Returns:
            Provider instance with computed UUID

        Example:
            >>> provider = Provider.create("openrouter")
            >>> provider.id  # deterministic UUID
            UUID('...')
        """
        provider_id = compute_provider_uuid(name)

        return cls(
            id=provider_id,
            name=name,
        )


class PromptTemplate(BaseModel):
    """PromptTemplate domain entity - the actual prompt text.

    Content-addressable - same template text always produces same UUID.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from template text hash"
    )

    template_text: str = Field(
        ...,
        description="Raw prompt template (e.g., Jinja template string)"
    )

    @classmethod
    def create(cls, template_text: str) -> "PromptTemplate":
        """Create a PromptTemplate with deterministic UUID.

        Args:
            template_text: Raw template string (before variable substitution)

        Returns:
            PromptTemplate instance with computed UUID (content-addressable)

        Example:
            >>> template = PromptTemplate.create("Query: {{ query }}\\nDoc: {{ document }}")
            >>> template.id  # deterministic UUID from hash
            UUID('...')
        """
        template_id = compute_prompt_template_uuid(template_text)

        return cls(
            id=template_id,
            template_text=template_text,
        )


class ModelSpec(BaseModel):
    """ModelSpec domain entity - model specification with inference parameters.

    Captures all experimental parameters that affect LLM behavior:
    - Which model (model_id)
    - Which provider
    - Inference parameters (temperature, top_p, etc.)

    This is NOT a config - configs are YAML files that point to adapter classes.
    This is a domain entity capturing what matters for experiments.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from spec name"
    )

    name: str = Field(
        ...,
        description="Spec name (e.g., 'gpt-oss-20b')"
    )

    model_id: str = Field(
        ...,
        description="Model identifier (e.g., 'gryphe/mythomax-l2-13b')"
    )

    provider: Provider = Field(
        ...,
        description="Provider entity"
    )

    context_window: int = Field(
        ...,
        gt=0,
        description="Maximum context window size in tokens"
    )

    # Inference parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    seed: Optional[int] = None

    # Additional parameters (JSONB in database)
    additional_params: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        model_id: str,
        provider: Provider,
        context_window: int,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> "ModelSpec":
        """Create a ModelSpec with deterministic UUID.

        Args:
            name: Spec name (e.g., "gpt-oss-20b")
            model_id: Model identifier
            provider: Provider entity
            context_window: Maximum context window size
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            seed: Random seed for reproducibility
            additional_params: Additional provider-specific parameters
            capabilities: Model capabilities metadata

        Returns:
            ModelSpec instance with computed UUID

        Example:
            >>> provider = Provider.create("openrouter")
            >>> spec = ModelSpec.create(
            ...     name="gpt-oss-20b",
            ...     model_id="gryphe/mythomax-l2-13b",
            ...     provider=provider,
            ...     context_window=8192,
            ...     temperature=0.7,
            ... )
            >>> spec.id  # deterministic UUID
            UUID('...')
        """
        spec_id = compute_model_spec_uuid(name)

        return cls(
            id=spec_id,
            name=name,
            model_id=model_id,
            provider=provider,
            context_window=context_window,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            additional_params=additional_params or {},
            capabilities=capabilities or {},
        )
