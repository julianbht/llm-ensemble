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
    models_created: int = Field(default=0, ge=0, description="Number of models created")
    models_skipped: int = Field(default=0, ge=0, description="Number of models skipped (already existed)")
    model_configs_created: int = Field(default=0, ge=0, description="Number of model configs created")
    model_configs_skipped: int = Field(default=0, ge=0, description="Number of model configs skipped (already existed)")
    prompt_templates_created: int = Field(default=0, ge=0, description="Number of prompt templates created")
    prompt_templates_skipped: int = Field(default=0, ge=0, description="Number of prompt templates skipped (already existed)")
    parser_created: int = Field(default=0, ge=0, description="Number of parser specs created")
    parser_skipped: int = Field(default=0, ge=0, description="Number of parser specs skipped (already existed)")
    infer_run_configs_created: int = Field(default=0, ge=0, description="Number of infer run configs created")
    infer_run_configs_skipped: int = Field(default=0, ge=0, description="Number of infer run configs skipped (already existed)")
    infer_runs_created: int = Field(default=0, ge=0, description="Number of infer runs created")
    infer_runs_skipped: int = Field(default=0, ge=0, description="Number of infer runs skipped (already existed)")
    infer_run_outputs_created: int = Field(default=0, ge=0, description="Number of infer run outputs created")
    infer_run_outputs_skipped: int = Field(default=0, ge=0, description="Number of infer run outputs skipped (already existed)")

    # Dataset finalization (created in close)
    judged_datasets_created: int = Field(default=0, ge=0, description="Number of judged datasets created")
    judged_datasets_skipped: int = Field(default=0, ge=0, description="Number of judged datasets skipped (already existed)")
    judged_dataset_junctions_created: int = Field(default=0, ge=0, description="Number of dataset-judgement junction records created")

    # Per-judgement entities (streamed during write_one)
    llm_prompts_created: int = Field(default=0, ge=0, description="Number of LLM prompt texts created")
    llm_prompts_skipped: int = Field(default=0, ge=0, description="Number of LLM prompt texts skipped (already existed)")
    llm_responses_created: int = Field(default=0, ge=0, description="Number of LLM response texts created")
    llm_responses_skipped: int = Field(default=0, ge=0, description="Number of LLM response texts skipped (already existed)")
    llm_invocation_metrics_created: int = Field(default=0, ge=0, description="Number of LLM invocation metrics created")
    llm_invocation_metrics_skipped: int = Field(default=0, ge=0, description="Number of LLM invocation metrics skipped (already existed)")
    llm_scores_created: int = Field(default=0, ge=0, description="Number of LLM scores created")
    llm_scores_skipped: int = Field(default=0, ge=0, description="Number of LLM scores skipped (already existed)")
    llm_judgements_created: int = Field(default=0, ge=0, description="Number of LLM judgements created")
    llm_judgements_skipped: int = Field(default=0, ge=0, description="Number of LLM judgements skipped (already existed)")

    def add_providers(self, created: int = 0, skipped: int = 0) -> None:
        """Increment provider counts."""
        self.providers_created += created
        self.providers_skipped += skipped

    def add_models(self, created: int = 0, skipped: int = 0) -> None:
        """Increment model counts."""
        self.models_created += created
        self.models_skipped += skipped

    def add_model_configs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment model config counts."""
        self.model_configs_created += created
        self.model_configs_skipped += skipped

    def add_prompt_templates(self, created: int = 0, skipped: int = 0) -> None:
        """Increment prompt template counts."""
        self.prompt_templates_created += created
        self.prompt_templates_skipped += skipped

    def add_parser(self, created: int = 0, skipped: int = 0) -> None:
        """Increment parser spec counts."""
        self.parser_created += created
        self.parser_skipped += skipped

    def add_infer_run_configs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment infer run config counts."""
        self.infer_run_configs_created += created
        self.infer_run_configs_skipped += skipped

    def add_infer_runs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment infer run counts."""
        self.infer_runs_created += created
        self.infer_runs_skipped += skipped

    def add_infer_run_outputs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment infer run output counts."""
        self.infer_run_outputs_created += created
        self.infer_run_outputs_skipped += skipped

    def add_llm_prompts(self, created: int = 0, skipped: int = 0) -> None:
        """Increment LLM prompt text counts."""
        self.llm_prompts_created += created
        self.llm_prompts_skipped += skipped

    def add_llm_responses(self, created: int = 0, skipped: int = 0) -> None:
        """Increment LLM response text counts."""
        self.llm_responses_created += created
        self.llm_responses_skipped += skipped

    def add_llm_invocation_metrics(self, created: int = 0, skipped: int = 0) -> None:
        """Increment LLM invocation metrics counts."""
        self.llm_invocation_metrics_created += created
        self.llm_invocation_metrics_skipped += skipped

    def add_llm_scores(self, created: int = 0, skipped: int = 0) -> None:
        """Increment LLM score counts."""
        self.llm_scores_created += created
        self.llm_scores_skipped += skipped

    def add_llm_judgements(self, created: int = 0, skipped: int = 0) -> None:
        """Increment LLM judgement counts."""
        self.llm_judgements_created += created
        self.llm_judgements_skipped += skipped

    def add_judged_datasets(self, created: int = 0, skipped: int = 0) -> None:
        """Increment judged dataset counts."""
        self.judged_datasets_created += created
        self.judged_datasets_skipped += skipped

    def add_judged_dataset_junctions(self, created: int = 0) -> None:
        """Increment judged dataset junction counts."""
        self.judged_dataset_junctions_created += created

    @property
    def total_created(self) -> int:
        """Total entities created across all types."""
        return (
            self.providers_created
            + self.models_created
            + self.model_configs_created
            + self.prompt_templates_created
            + self.parser_created
            + self.infer_run_configs_created
            + self.infer_runs_created
            + self.infer_run_outputs_created
            + self.judged_datasets_created
            + self.judged_dataset_junctions_created
            + self.llm_prompts_created
            + self.llm_responses_created
            + self.llm_invocation_metrics_created
            + self.llm_scores_created
            + self.llm_judgements_created
        )

    @property
    def total_skipped(self) -> int:
        """Total entities skipped across all types."""
        return (
            self.providers_skipped
            + self.models_skipped
            + self.model_configs_skipped
            + self.prompt_templates_skipped
            + self.parser_skipped
            + self.infer_run_configs_skipped
            + self.infer_runs_skipped
            + self.infer_run_outputs_skipped
            + self.judged_datasets_skipped
            + self.llm_prompts_skipped
            + self.llm_responses_skipped
            + self.llm_invocation_metrics_skipped
            + self.llm_scores_skipped
            + self.llm_judgements_skipped
        )
