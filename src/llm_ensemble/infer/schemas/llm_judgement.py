"""LLMJudgement schema - complete LLM relevance judgement with full provenance.

This is the canonical output schema for the infer CLI, representing a complete
LLM judgement with full data lineage (input sample + LLM response + parsed score + infer run info).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.infer.schemas.llm_request import LLMRequest
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas.llm_score import LLMScore
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.warnings import BaseWarning


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement with full provenance.

    This is the canonical judgement schema that combines:
    - sample: The input (query + document + gold score + ingest manifest)
    - llm_request: What we sent to the LLM (rendered prompt + prompt warnings)
    - llm_response: The raw LLM output (unparsed text + observability metadata + provider warnings)
    - llm_score: The parsed relevance assessment (label + confidence + rationale + parser warnings)
    - run_info: The inference run context (model config + prompt config + git info + input params)

    This captures the complete data lineage: what was judged, what prompt was sent,
    what response came back, and what score was extracted. Each stage has its own warnings.

    Many LLMJudgements share one InferRunInfo (Many-to-One relationship). The run_info
    contains immutable runtime context known before the run starts, allowing judgements
    to be serialized immediately without waiting for aggregate statistics.
    """

    judging_sample: JudgingSample = Field(
        ...,
        description="The input sample that was judged (includes ingest manifest)"
    )

    llm_request: LLMRequest = Field(
        ...,
        description="The request sent to the LLM (rendered prompt + prompt warnings)"
    )

    llm_response: LLMResponse = Field(
        ...,
        description="The raw LLM output (unparsed text + observability metadata + provider warnings)"
    )

    llm_score: Optional[LLMScore] = Field(
        None,
        description=(
            "The parsed relevance assessment (label + confidence + rationale). "
            "None if response parsing completely failed."
        )
    )

    run_info: InferRunInfo = Field(
        ...,
        description="Inference run context (Many-to-One: many judgements share one run_info)"
    )

    def get_all_warnings(self) -> list[BaseWarning]:
        """Aggregate all warnings from request, response, and score.

        Combines warnings from all three stages:
        - Prompt building (in llm_request)
        - LLM inference (in llm_response)
        - Response parsing (in llm_score)

        Returns:
            List of all warnings from this judgement (prompt + provider + parser)

        Example:
            >>> judgement = LLMJudgement(...)
            >>> all_warnings = judgement.get_all_warnings()
            >>> len(all_warnings)  # Total warnings from all stages
            5
        """
        warnings = []

        # Prompt warnings from llm_request
        warnings.extend(self.llm_request.warnings)

        # Provider warnings from llm_response
        warnings.extend(self.llm_response.warnings)

        # Parser warnings from llm_score (if score exists)
        if self.llm_score is not None:
            warnings.extend(self.llm_score.warnings)

        return warnings
