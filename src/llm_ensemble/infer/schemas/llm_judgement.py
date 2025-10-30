"""LLMJudgement schema - complete LLM relevance judgement with full provenance.

This is the canonical output schema for the infer CLI, representing a complete
LLM judgement with full data lineage (input sample + LLM response + parsed score + infer manifest).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas.llm_score import LLMScore
from llm_ensemble.infer.schemas.infer_manifest import InferManifest
from llm_ensemble.infer.schemas.warnings import BaseWarning


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement with full provenance.

    This is the canonical judgement schema that combines:
    - sample: The input (query + document + gold score + ingest manifest)
    - llm_response: The raw LLM output (unparsed text + observability metadata)
    - llm_score: The parsed relevance assessment (label + confidence + rationale)
    - manifest: The inference run manifest (model config + timing + git info)

    This captures the complete data lineage: what was judged, how it was judged,
    what the raw response was, and what structured score was extracted.

    Many LLMJudgements share one InferManifest (Many-to-One relationship).
    """

    sample: JudgingSample = Field(
        ...,
        description="The input sample that was judged (includes ingest manifest)"
    )

    llm_response: LLMResponse = Field(
        ...,
        description="The raw LLM output (unparsed text + observability metadata)"
    )

    llm_score: Optional[LLMScore] = Field(
        None,
        description=(
            "The parsed relevance assessment (label + confidence + rationale). "
            "None if response parsing completely failed."
        )
    )

    manifest: InferManifest = Field(
        ...,
        description="Inference run manifest (Many-to-One: many judgements share one manifest)"
    )

    def get_all_warnings(self) -> list[BaseWarning]:
        """Aggregate all warnings from response and score.

        Combines warnings from both the provider (in llm_response) and
        parser (in llm_score) into a single list for collector aggregation.

        Returns:
            List of all warnings from this judgement (provider + parser)

        Example:
            >>> judgement = LLMJudgement(...)
            >>> all_warnings = judgement.get_all_warnings()
            >>> len(all_warnings)  # Total warnings from provider + parser
            3
        """
        warnings = []

        # Provider warnings from llm_response
        warnings.extend(self.llm_response.warnings)

        # Parser warnings from llm_score (if score exists)
        if self.llm_score is not None:
            warnings.extend(self.llm_score.warnings)

        return warnings
