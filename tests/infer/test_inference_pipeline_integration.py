"""Integration tests for InferenceApplication with real domain adapters.

This test module demonstrates end-to-end pipeline testing with:
- REAL domain logic: actual parsers, prompt builders (business rules)
- MOCK infrastructure: LLM API calls, file I/O (external dependencies)

These tests verify that the complete inference pipeline works correctly
with actual production adapters, while avoiding slow external calls.

Difference from test_inference_application.py:
- That file: Tests orchestration logic in isolation (all mocks)
- This file: Tests real pipeline integration (hybrid mocks + real adapters)

These are NOT unit tests for adapters (those exist separately).
These verify that real components integrate correctly within the pipeline.
"""

from __future__ import annotations
import pytest
from pathlib import Path

from llm_ensemble.infer.application.inference_application import InferenceApplication
from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_parser import ThomasAdvancedParser
from llm_ensemble.infer.adapters.driven.prompts.thomas_advanced_prompt_builder import ThomasAdvancedPromptBuilder
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore

from tests.infer.mocks import (
    MockInputAdapter,
    MockOutputAdapter,
    MockLLMProvider,
)


@pytest.mark.integration
def test_inference_pipeline_slice(
    sample_dataset_five: NormalizedDataset,
    temp_run_dir: Path
):
    """Test complete inference pipeline with real parser and prompt builder.

    Uses REAL adapters:
    - ThomasAdvancedPromptBuilder (real template rendering)
    - ThomasAdvancedParser (real multi-stage parsing: extract -> validate -> map)

    Uses MOCK infrastructure:
    - MockLLMProvider (no actual API calls)
    - MockInputAdapter/MockOutputAdapter (no actual file I/O)

    Verifies:
    - Dataset slicing works correctly with real adapters
    - Only samples within [start_idx:end_idx) are processed
    - Real prompt builder creates valid prompts from dataset samples
    - Real parser correctly extracts scores from LLM response format
    - Run summary metrics are calculated correctly
    """
    # Arrange: Real domain adapters + mock infrastructure
    output_adapter = MockOutputAdapter()
    app = InferenceApplication(
        input_port=MockInputAdapter(sample_dataset_five),
        output_port=output_adapter,
        prompt_builder=ThomasAdvancedPromptBuilder(), 
        llm_provider=MockLLMProvider(mock_response='{"M": 2, "T": 1, "O": 1}'), 
        response_parser=ThomasAdvancedParser(),
        run_dir=temp_run_dir,
        run_name="integration-test-run"
    )

    # Act: process samples at indices 2 and 3
    summary = app.run_inference(
        input_run_name="test-ingest-run",
        start_idx=2,
        end_idx=4,
        official=False,
        notes="Integration test with real adapters"
    )

    # Assert: Only sliced samples were processed
    assert summary.judgements.total_count == 2
    assert len(output_adapter.written_judgements) == 2

    # Assert: Correct samples from dataset were processed
    first_judgement = output_adapter.written_judgements[0]
    second_judgement = output_adapter.written_judgements[1]
    assert first_judgement.dataset_sample == sample_dataset_five.samples[2]
    assert first_judgement.dataset_sample.sequence_number == 2
    assert second_judgement.dataset_sample == sample_dataset_five.samples[3]
    assert second_judgement.dataset_sample.sequence_number == 3

    # Assert: Real parser successfully extracted scores from both samples
    assert summary.judgements.failed_parses_count == 0
    assert first_judgement.llm_score is not None
    assert first_judgement.llm_score.label == RelevanceScore.RELEVANT
    assert second_judgement.llm_score is not None
    assert second_judgement.llm_score.label == RelevanceScore.RELEVANT

    # Assert: Run summary metrics calculated correctly
    assert summary.persistence.total_created == 2
