"""Unit tests for Cohen's Kappa metric adapter.

Tests validate implementation against LLM Judge Challenge official results.
"""

import pytest

from llm_ensemble.evaluate.adapters.driven.metrics.cohens_kappa import CohensKappaAdapter
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore

from .test_helpers import (
    load_qrels,
    load_submission,
    get_expected_metric,
)


@pytest.mark.unit
class TestCohensKappaAdapter:
    """Test suite for Cohen's Kappa adapter."""

    def test_perfect_agreement(self):
        """Test perfect agreement returns kappa = 1.0."""
        adapter = CohensKappaAdapter()
        ground_truth = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT
        ]
        predictions = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT
        ]

        result = adapter.compute(ground_truth, predictions)

        assert result.value == pytest.approx(1.0, abs=1e-6)
        assert result.name == "cohens_kappa"
        assert result.sample_size == 3

    def test_complete_disagreement(self):
        """Test systematic disagreement returns kappa close to zero."""
        adapter = CohensKappaAdapter()
        ground_truth = [RelevanceScore.IRRELEVANT] * 10
        predictions = [RelevanceScore.PERFECTLY_RELEVANT] * 10

        result = adapter.compute(ground_truth, predictions)

        assert result.value == pytest.approx(0.0, abs=0.01)

    def test_handles_none_predictions(self):
        """Test missing predictions (None values) are filtered out."""
        adapter = CohensKappaAdapter()
        ground_truth = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT,
            RelevanceScore.PERFECTLY_RELEVANT
        ]
        predictions = [
            RelevanceScore.IRRELEVANT,
            None,  # Parse failed
            RelevanceScore.HIGHLY_RELEVANT,
            None   # Parse failed
        ]

        result = adapter.compute(ground_truth, predictions)

        # Kappa should be computed using only valid pairs (2 out of 4)
        assert result.sample_size == 2
        assert result.value == pytest.approx(1.0)  # Both valid pairs match perfectly

    def test_all_none_predictions(self):
        """Test all None predictions raises ValueError."""
        adapter = CohensKappaAdapter()
        ground_truth = [RelevanceScore.IRRELEVANT, RelevanceScore.RELEVANT]
        predictions = [None, None]

        with pytest.raises(ValueError, match="all predictions are None"):
            adapter.compute(ground_truth, predictions)

    def test_too_many_missing_predictions(self):
        """Test more than 5 missing predictions raises ValueError."""
        adapter = CohensKappaAdapter()
        ground_truth = [RelevanceScore.IRRELEVANT] * 10
        predictions = [None] * 6 + [RelevanceScore.RELEVANT] * 4

        with pytest.raises(ValueError, match="Too many missing predictions"):
            adapter.compute(ground_truth, predictions)

    def test_empty_input_raises_error(self):
        """Test empty input raises ValueError."""
        adapter = CohensKappaAdapter()

        with pytest.raises(ValueError, match="empty input"):
            adapter.compute([], [])

    def test_mismatched_lengths_raises_error(self):
        """Test mismatched input lengths raise ValueError."""
        adapter = CohensKappaAdapter()
        ground_truth = [RelevanceScore.IRRELEVANT]
        predictions = [RelevanceScore.RELEVANT, RelevanceScore.HIGHLY_RELEVANT]

        with pytest.raises(ValueError, match="same length"):
            adapter.compute(ground_truth, predictions)



@pytest.mark.unit
class TestCohensKappaWithChalllengeData:
    """Validate Cohen's Kappa implementation against LLM Judge Challenge official results.

    These tests use real submission data and ground truth from the challenge
    to ensure our implementation produces identical results to the official metrics.
    """

    def test_trema_direct_submission(self):
        """Test TREMA-direct submission matches official Cohen's Kappa."""
        adapter = CohensKappaAdapter()
        ground_truth = load_qrels()
        predictions = load_submission("TREMA-direct.txt")
        expected_kappa = get_expected_metric("TREMA-direct", "cohenskappa")

        result = adapter.compute(ground_truth, predictions)

        assert result.value == pytest.approx(expected_kappa, abs=0.0001)
        assert result.sample_size == 4423  # Full dataset
        assert result.name == "cohens_kappa"

    def test_h2oloo_zeroshot1_submission(self):
        """Test h2oloo-zeroshot1 submission matches official Cohen's Kappa."""
        adapter = CohensKappaAdapter()
        ground_truth = load_qrels()
        predictions = load_submission("h2oloo-zeroshot1.txt")
        expected_kappa = get_expected_metric("h2oloo-zeroshot1", "cohenskappa")

        result = adapter.compute(ground_truth, predictions)

        assert result.value == pytest.approx(expected_kappa, abs=0.0001)
        assert result.sample_size == 4423

    def test_rmitir_llama70b_submission(self):
        """Test RMITIR-llama70B submission matches official Cohen's Kappa."""
        adapter = CohensKappaAdapter()
        ground_truth = load_qrels()
        predictions = load_submission("RMITIR-llama70B.txt")
        expected_kappa = get_expected_metric("RMITIR-llama70B", "cohenskappa")

        result = adapter.compute(ground_truth, predictions)

        assert result.value == pytest.approx(expected_kappa, abs=0.0001)
        assert result.sample_size == 4423

    def test_willia_umbrela1_submission(self):
        """Test willia-umbrela1 submission matches official Cohen's Kappa."""
        adapter = CohensKappaAdapter()
        ground_truth = load_qrels()
        predictions = load_submission("willia-umbrela1.txt")
        expected_kappa = get_expected_metric("willia-umbrela1", "cohenskappa")

        result = adapter.compute(ground_truth, predictions)

        assert result.value == pytest.approx(expected_kappa, abs=0.0001)
        assert result.sample_size == 4423

    @pytest.mark.parametrize("submission_name", [
        "NISTRetrieval-instruct0.txt",
        "Olz-gpt4o.txt",
        "TREMA-4prompts.txt",
        "prophet-setting1.txt",
    ])
    def test_multiple_submissions_match_official_results(self, submission_name: str):
        """Test multiple submissions match official Cohen's Kappa values.

        This parametrized test validates our implementation across diverse
        submission types to ensure correctness.
        """
        adapter = CohensKappaAdapter()
        ground_truth = load_qrels()
        predictions = load_submission(submission_name)

        # Extract submission ID (remove .txt extension)
        submission_id = submission_name.replace(".txt", "")
        expected_kappa = get_expected_metric(submission_id, "cohenskappa")

        result = adapter.compute(ground_truth, predictions)

        assert result.value == pytest.approx(expected_kappa, abs=0.0001), (
            f"Cohen's Kappa mismatch for {submission_id}: "
            f"computed={result.value:.4f}, expected={expected_kappa:.4f}"
        )
