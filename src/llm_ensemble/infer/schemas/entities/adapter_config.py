"""AdapterConfig entity for the infer CLI.

Configuration bundle for inference adapters (prompt builder, parser, provider).
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.schemas.entities.parser import Parser
from llm_ensemble.infer.schemas.entities.provider import Provider


class AdapterConfig(BaseModel):
    """Adapter configuration bundle.

    Bundles the three adapter components used during inference:
    - prompt_builder: Which prompt template and builder logic (includes template_text)
    - parser: Which response parser
    - provider: Which LLM provider service

    This represents the adapter configuration used to produce judgements,
    separate from the model configuration (which lives on JudgedDataset).

    By bundling full entities (not just names), the writer can access all metadata
    needed for ORM upserts (like template_text from PromptBuilder).
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this adapter config"
    )

    prompt_builder: PromptBuilder = Field(
        ...,
        description="Full prompt builder entity (includes name and template_text)"
    )

    parser: Parser = Field(
        ...,
        description="Full parser entity (includes name)"
    )

    provider: Provider = Field(
        ...,
        description="Full provider entity (includes name)"
    )

    @property
    def prompt_builder_name(self) -> str:
        """Get prompt builder name for convenience."""
        return self.prompt_builder.name

    @property
    def parser_name(self) -> str:
        """Get parser name for convenience."""
        return self.parser.name

    @property
    def provider_name(self) -> str:
        """Get provider name for convenience."""
        return self.provider.name
