from __future__ import annotations
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field


class ModelJudgement(BaseModel):
    """Output from a single LLM judge for one query-document pair.

    This is the canonical judgement schema that all adapters must produce.
    Follows the data contract specified in CLAUDE.md.
    """

    # Identity
    model_id: str = Field(..., description="Model identifier (e.g., 'phi3-mini')")
    provider: str = Field(..., description="Provider name (e.g., 'hf', 'ollama')")
    version: Optional[str] = Field(None, description="Model version if available")

    # Input identifiers (from JudgingExample)
    query_id: str = Field(..., description="Query ID from input")
    docid: str = Field(..., description="Document ID from input")

    # Core judgement output
    label: Optional[Literal[0, 1, 2]] = Field(
        None,
        description=(
            "Relevance label: "
            "2 = highly relevant, very helpful for this query; "
            "1 = relevant, may be partly helpful; "
            "0 = not relevant, should never be shown for this query. "
            "None = parsing failed or model did not produce valid output."
        )
    )
    score: Optional[float] = Field(None, ge=0.0, le=2.0, description="Relevance score [0-2]")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model self-reported or derived uncertainty")

    # Explainability
    rationale: Optional[str] = Field(None, description="Model's explanation for its judgement")
    raw_text: str = Field(..., description="Unparsed model output for debugging")

    # Observability
    latency_ms: float = Field(..., description="Inference time in milliseconds")
    retries: int = Field(0, description="Number of retries attempted")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")

    # Warnings/metadata
    warnings: list[str] = Field(default_factory=list, description="Parser warnings, fallbacks, etc.")


class ModelConfig(BaseModel):
    """Domain model for model configuration (mirrors configs/models/*.yaml)."""

    model_id: str
    provider: Literal["hf", "ollama", "openrouter"]
    context_window: int
    default_params: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)

    # HuggingFace-specific fields
    hf_endpoint_url: Optional[str] = Field(None, description="HF Inference Endpoint URL")
    hf_model_name: Optional[str] = Field(None, description="HF model repo name")

    # OpenRouter-specific fields
    openrouter_model_id: Optional[str] = Field(None, description="OpenRouter model ID")


class PromptConfig(BaseModel):
    """Configuration for a prompt template.

    Specifies which variables to pass to the template and any metadata
    about the prompt's purpose and expected outputs.
    """

    name: str = Field(..., description="Prompt identifier (matches filename without .jinja)")
    template_file: str = Field(..., description="Template filename (e.g., 'thomas-et-al-prompt.jinja')")
    description: Optional[str] = Field(None, description="Human-readable description of the prompt")

    # Variables to pass to the Jinja2 template
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to pass when rendering the template (e.g., role=True, aspects=False)"
    )

    # Metadata about expected outputs
    expected_output_format: Optional[str] = Field(
        None,
        description="Expected format of model response (e.g., 'json', 'text')"
    )
    response_parser: Optional[str] = Field(
        None,
        description="Name of parser function to use (e.g., 'parse_thomas_response')"
    )
