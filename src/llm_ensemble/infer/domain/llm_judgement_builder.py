"""Builder for LLMJudgement entity.

Domain Layer - Builder Pattern

Incrementally builds LLMJudgement as the inference pipeline progresses.
Each builder method corresponds to a pipeline step, making the flow explicit.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.ingest.domain.entities.dataset_sample import DatasetSample
from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssue


class LLMJudgementBuilder:
    """Builder for LLMJudgement that mirrors the inference pipeline flow.

    Incrementally builds a judgement as each step of inference completes:
    1. Start with the dataset sample being judged
    2. Add the prompt after building it
    3. Add the LLM response and metrics after inference
    4. Add the parsed score and warnings after parsing
    5. Build the final judgement

    This makes the pipeline flow explicit in the code.
    """

    def __init__(self, dataset_sample: DatasetSample):
        """Initialize builder with the sample being judged.

        Args:
            dataset_sample: The query-document pair being judged
        """
        self._dataset_sample = dataset_sample
        self._prompt_text: Optional[str] = None
        self._response_text: Optional[str] = None
        self._llm_invocation_metrics: Optional[LLMInvocationMetrics] = None
        self._llm_score: Optional[LLMScore] = None
        self._parse_issues: list[ParserIssue] = []

    def with_prompt(self, prompt_text: str) -> "LLMJudgementBuilder":
        """Add the prompt text (called after prompt building).

        Args:
            prompt_text: The rendered prompt sent to the LLM

        Returns:
            Self for method chaining
        """
        self._prompt_text = prompt_text
        return self

    def with_llm_response(
        self,
        response_text: str,
        invocation_metrics: LLMInvocationMetrics
    ) -> "LLMJudgementBuilder":
        """Add the LLM response and metrics (called after inference).

        Args:
            response_text: Raw response text from the LLM
            invocation_metrics: Performance metrics (latency, cost, tokens)

        Returns:
            Self for method chaining
        """
        self._response_text = response_text
        self._llm_invocation_metrics = invocation_metrics
        return self

    def with_parsed_score(
        self,
        llm_score: Optional[LLMScore],
        parse_issues: list[ParserIssue]
    ) -> "LLMJudgementBuilder":
        """Add the parsed score and warnings (called after parsing).

        Args:
            llm_score: Parsed score (may be None if parsing failed)
            parse_issues: Any warnings from the parsing process

        Returns:
            Self for method chaining
        """
        self._llm_score = llm_score
        self._parse_issues = parse_issues
        return self

    def build(self) -> LLMJudgement:
        """Build the final LLMJudgement entity.

        Returns:
            Complete LLMJudgement entity

        Raises:
            ValueError: If required fields are missing
        """
        if self._prompt_text is None:
            raise ValueError("prompt_text is required - call with_prompt() first")
        if self._response_text is None:
            raise ValueError("response_text is required - call with_llm_response() first")
        if self._llm_invocation_metrics is None:
            raise ValueError("llm_invocation_metrics is required - call with_llm_response() first")

        return LLMJudgement(
            dataset_sample=self._dataset_sample,
            prompt_text=self._prompt_text,
            response_text=self._response_text,
            llm_invocation_metrics=self._llm_invocation_metrics,
            llm_score=self._llm_score,
            parse_issues=self._parse_issues,
        )
