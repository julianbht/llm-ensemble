"""LLMJudgement schema - complete LLM relevance judgement with full provenance.

This is the canonical output schema for the infer CLI, representing a complete
LLM judgement with full data lineage (input sample + LLM response + infer manifest).
"""

from __future__ import annotations
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas.infer_manifest import InferManifest


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement with full provenance.

    This is the canonical judgement schema that combines:
    - sample: The input (query + document + gold score + ingest manifest)
    - llm_response: The LLM's output (predicted score + rationale + metadata)
    - manifest: The inference run manifest (model config + timing + git info)

    This captures the complete data lineage: what was judged, how it was judged,
    and what the judgement was.

    Many LLMJudgements share one InferManifest (Many-to-One relationship).
    """

    sample: JudgingSample = Field(
        ...,
        description="The input sample that was judged (includes ingest manifest)"
    )

    llm_response: LLMResponse = Field(
        ...,
        description="The LLM's response (score, rationale, observability metadata)"
    )

    manifest: InferManifest = Field(
        ...,
        description="Inference run manifest (Many-to-One: many judgements share one manifest)"
    )
