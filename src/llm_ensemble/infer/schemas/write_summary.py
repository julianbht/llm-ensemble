"""Write summary schema for tracking judgement write operations.

Mutable builder for tracking what entities were persisted during inference writes.
Used as metadata in run summaries for reproducibility and debugging.

For per-write feedback, use WriteResult instead.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict


class WriteSummary(BaseModel):
    """Incremental builder for tracking write operations.

    Mutable object that tracks what entities were created vs. skipped.
    Writers add to it incrementally as each entity type is persisted.
    Used as metadata in run summaries - NOT for logging (adapters log directly).

    For per-write feedback with item IDs, use WriteResult instead.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Run metadata (created once during open)
    providers_created: int = Field(default=0, ge=0, description="Number of providers created")
    providers_skipped: int = Field(default=0, ge=0, description="Number of providers skipped (already existed)")
    model_specs_created: int = Field(default=0, ge=0, description="Number of model specs created")
    model_specs_skipped: int = Field(default=0, ge=0, description="Number of model specs skipped (already existed)")
    prompt_templates_created: int = Field(default=0, ge=0, description="Number of prompt templates created")
    prompt_templates_skipped: int = Field(default=0, ge=0, description="Number of prompt templates skipped (already existed)")
    parser_specs_created: int = Field(default=0, ge=0, description="Number of parser specs created")
    parser_specs_skipped: int = Field(default=0, ge=0, description="Number of parser specs skipped (already existed)")
    infer_runs_created: int = Field(default=0, ge=0, description="Number of infer runs created")
    infer_runs_skipped: int = Field(default=0, ge=0, description="Number of infer runs skipped (already existed)")

    # Per-judgement entities (streamed during write_one)
    llm_requests_created: int = Field(default=0, ge=0, description="Number of LLM requests created")
    llm_requests_skipped: int = Field(default=0, ge=0, description="Number of LLM requests skipped (already existed)")
    llm_scores_created: int = Field(default=0, ge=0, description="Number of LLM scores created")
    llm_scores_skipped: int = Field(default=0, ge=0, description="Number of LLM scores skipped (already existed)")
    llm_calls_created: int = Field(default=0, ge=0, description="Number of LLM calls created (joins request+score)")

    def add_providers(self, created: int = 0, skipped: int = 0) -> None:
        """Increment provider counts."""
        self.providers_created += created
        self.providers_skipped += skipped

    def add_model_specs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment model spec counts."""
        self.model_specs_created += created
        self.model_specs_skipped += skipped

    def add_prompt_templates(self, created: int = 0, skipped: int = 0) -> None:
        """Increment prompt template counts."""
        self.prompt_templates_created += created
        self.prompt_templates_skipped += skipped

    def add_parser_specs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment parser spec counts."""
        self.parser_specs_created += created
        self.parser_specs_skipped += skipped

    def add_infer_runs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment infer run counts."""
        self.infer_runs_created += created
        self.infer_runs_skipped += skipped

    def add_llm_requests(self, created: int = 0, skipped: int = 0) -> None:
        """Increment LLM request counts."""
        self.llm_requests_created += created
        self.llm_requests_skipped += skipped

    def add_llm_scores(self, created: int = 0, skipped: int = 0) -> None:
        """Increment LLM score counts."""
        self.llm_scores_created += created
        self.llm_scores_skipped += skipped

    def add_llm_calls(self, created: int = 1) -> None:
        """Increment LLM call count (always created, never skipped)."""
        self.llm_calls_created += created

    @property
    def total_created(self) -> int:
        """Total entities created across all types."""
        return (
            self.providers_created
            + self.model_specs_created
            + self.prompt_templates_created
            + self.parser_specs_created
            + self.infer_runs_created
            + self.llm_requests_created
            + self.llm_scores_created
            + self.llm_calls_created
        )

    @property
    def total_skipped(self) -> int:
        """Total entities skipped across all types."""
        return (
            self.providers_skipped
            + self.model_specs_skipped
            + self.prompt_templates_skipped
            + self.parser_specs_skipped
            + self.infer_runs_skipped
            + self.llm_requests_skipped
            + self.llm_scores_skipped
        )
