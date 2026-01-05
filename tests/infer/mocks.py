"""Mock adapter implementations for testing the infer pipeline.

This module provides test doubles (mocks) for all driven ports in the infer CLI.
These mocks replace real infrastructure (file I/O, API calls, databases) with
in-memory implementations, enabling fast and deterministic testing of business logic.

Mock adapters are reusable across the infer test suite.
"""

from __future__ import annotations
from typing import Optional
from uuid import uuid4

from llm_ensemble.infer.application.ports.driven.for_input import ForInput
from llm_ensemble.infer.application.ports.driven.for_output import ForOutput
from llm_ensemble.infer.application.ports.driven.for_building_prompts import ForBuildingPrompts
from llm_ensemble.infer.application.ports.driven.for_invoking_llm import ForInvokingLLM
from llm_ensemble.infer.application.ports.driven.for_parsing_responses import ForParsingResponses
from llm_ensemble.infer.application.write_summary import WriteSummary
from llm_ensemble.infer.domain.entities.infer_run import InferRun
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement, LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssue
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class MockInputAdapter(ForInput):
    """Mock input adapter that returns predefined test data.

    Replaces real file/database readers with in-memory test data.
    Allows testing without file I/O dependencies.
    """

    def __init__(self, mock_dataset: NormalizedDataset):
        """Initialize with test dataset.

        Args:
            mock_dataset: Predefined dataset to return
        """
        self.mock_dataset = mock_dataset
        self.read_called = False
        self.read_call_args: dict[str, Optional[str | int]] = {}

    def read(self, run_name: str, limit: Optional[int] = None) -> NormalizedDataset:
        """Return mock dataset and track call."""
        self.read_called = True
        self.read_call_args = {"run_name": run_name, "limit": limit}
        return self.mock_dataset


class MockOutputAdapter(ForOutput):
    """Mock output adapter that captures written judgements in memory.

    Replaces real file/database writers with in-memory collection.
    Allows verification of what was written without actual I/O.
    """

    def __init__(self):
        """Initialize empty collections for tracking writes."""
        self.written_judgements: list[LLMJudgement] = []
        self.infer_run: Optional[InferRun] = None
        self.is_open = False
        self._write_summary = WriteSummary()

    @property
    def io_name(self) -> str:
        """Return adapter name."""
        return "mock"

    def open(self, infer_run: InferRun) -> "MockOutputAdapter":
        """Track InferRun and mark as open."""
        self.infer_run = infer_run
        self.is_open = True
        return self

    def write_one(self, judgement: LLMJudgement) -> None:
        """Capture judgement in memory."""
        if not self.is_open:
            raise RuntimeError("Writer not open")
        self.written_judgements.append(judgement)
        self._write_summary.add_llm_judgements(created=1)

    def close(self) -> WriteSummary:
        """Mark as closed and return summary."""
        self.is_open = False
        return self._write_summary

    def get_write_summary(self) -> WriteSummary:
        """Return write summary."""
        return self._write_summary


class MockPromptBuilder(ForBuildingPrompts):
    """Mock prompt builder that returns deterministic prompts.

    Replaces real template rendering with simple string formatting.
    Allows testing without template engine dependencies.
    """

    def __init__(self):
        """Initialize with empty call tracking."""
        self.build_prompt_calls: list[NormalizedDatasetJudgingSample] = []
        self._builder = PromptBuilder(
            id=uuid4(),
            name="mock-builder",
            version="1.0"
        )
        self._template_text = "Query: {query}\nDocument: {document}\nRelevance?"

    def build_prompt(self, dataset_sample: NormalizedDatasetJudgingSample) -> str:
        """Build deterministic prompt and track call."""
        self.build_prompt_calls.append(dataset_sample)
        return (
            f"Query: {dataset_sample.judging_sample.query.query_text}\n"
            f"Document: {dataset_sample.judging_sample.document.doc_text}\n"
            "Rate relevance (0=irrelevant, 1=relevant, 2=highly relevant)"
        )

    def get_builder(self) -> PromptBuilder:
        """Return builder metadata."""
        return self._builder

    def get_template_text(self) -> str:
        """Return template text."""
        return self._template_text


class MockLLMProvider(ForInvokingLLM):
    """Mock LLM provider that returns predefined responses.

    Replaces real API calls with instant, deterministic responses.
    Allows testing without network I/O or API credentials.
    """

    def __init__(self, mock_response: str = '{"M": 2, "T": 1, "O": 1}'):
        """Initialize with predefined response.

        Args:
            mock_response: Response text to return for all infer() calls
        """
        self.mock_response = mock_response
        self.infer_calls: list[str] = []
        self._provider = Provider(name="mock-provider", version="1.0")
        self._model_config = ModelConfig(
            name="mock-model",
            name_hint="mock-model",
            model_id="mock/model",
            context_window=4096
        )
        self._retry_config = RetryConfig(
            name="mock-retry",
            max_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0
        )

    def infer(self, prompt: str) -> tuple[str, LLMInvocationMetrics]:
        """Return mock response and metrics, track call."""
        self.infer_calls.append(prompt)
        metrics = LLMInvocationMetrics(
            latency_ms=100.0,
            retries=0,
            cost_estimate_usd=0.001,
            actual_cost_usd=None,
            generation_id=None,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70
        )
        return self.mock_response, metrics

    def get_provider(self) -> Provider:
        """Return provider metadata."""
        return self._provider

    def get_model_config(self) -> ModelConfig:
        """Return model config."""
        return self._model_config

    def get_retry_config(self) -> RetryConfig:
        """Return retry config."""
        return self._retry_config


class MockResponseParser(ForParsingResponses):
    """Mock response parser that returns predefined scores.

    Replaces real parsing logic with deterministic score extraction.
    Allows testing without complex parsing dependencies.
    """

    def __init__(self, mock_score: Optional[LLMScore] = "default"):
        """Initialize with predefined score.

        Args:
            mock_score: Score to return for all parse() calls.
                       Use "default" for success case, None for parse failure.
        """
        if mock_score == "default":
            self.mock_score: Optional[LLMScore] = LLMScore(
                label=RelevanceScore.RELEVANT,
                confidence=None,
                rationale=None
            )
        else:
            self.mock_score = mock_score
        self.parse_calls: list[str] = []
        self._parser = ResponseParser(
            id=uuid4(),
            name="mock-parser",
            version="1.0"
        )

    def parse(self, raw_text: str) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Return mock score and track call."""
        self.parse_calls.append(raw_text)
        return self.mock_score, None

    def get_parser(self) -> ResponseParser:
        """Return parser metadata."""
        return self._parser
