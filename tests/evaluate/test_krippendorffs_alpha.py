"""Unit tests for Krippendorff's Alpha metric adapter."""

import pytest
import numpy as np

from llm_ensemble.evaluate.adapters.driven.metrics.krippendorffs_alpha import KrippendorffsAlphaAdapter
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


@pytest.mark.unit
class TestKrippendorffsAlphaAdapter:
    """Test suite for Krippendorff's Alpha adapter."""

    def test_perfect_agreement(self):
        """Test perfect agreement returns alpha = 1.0."""
        adapter = KrippendorffsAlphaAdapter()
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
        assert result.name == "krippendorffs_alpha"
        assert result.sample_size == 3
        assert "strong" in result.interpretation

    def test_complete_disagreement(self):
        """Test systematic disagreement returns negative alpha."""
        adapter = KrippendorffsAlphaAdapter()
        ground_truth = [RelevanceScore.IRRELEVANT] * 10
        predictions = [RelevanceScore.PERFECTLY_RELEVANT] * 10

        result = adapter.compute(ground_truth, predictions)

        assert result.value < 0
        assert "poor" in result.interpretation or "disagreement" in result.interpretation

    def test_handles_none_predictions(self):
        """Test missing predictions (None values) are handled correctly."""
        adapter = KrippendorffsAlphaAdapter()
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

        # Alpha should be computed using only valid pairs (2 out of 4)
        assert result.sample_size == 2
        assert not np.isnan(result.value)
        assert isinstance(result.interpretation, str)

    def test_all_none_predictions(self):
        """Test all None predictions raises ValueError."""
        adapter = KrippendorffsAlphaAdapter()
        ground_truth = [RelevanceScore.IRRELEVANT, RelevanceScore.RELEVANT]
        predictions = [None, None]

        with pytest.raises(ValueError, match="all predictions are None"):
            adapter.compute(ground_truth, predictions)

    def test_ordinal_scale_respected(self):
        """Test ordinal distances are computed correctly."""
        adapter = KrippendorffsAlphaAdapter()

        # Create test data with mix of agreement and disagreement
        # Case 1: Some perfect matches, some off by 1
        ground_truth_1 = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT,
            RelevanceScore.PERFECTLY_RELEVANT,
        ] * 3
        predictions_1 = [
            RelevanceScore.IRRELEVANT,  # Perfect match
            RelevanceScore.HIGHLY_RELEVANT,  # Off by 1
            RelevanceScore.HIGHLY_RELEVANT,  # Perfect match
            RelevanceScore.PERFECTLY_RELEVANT,  # Perfect match
        ] * 3

        # Case 2: Same agreements, but disagreements are off by 2 instead of 1
        ground_truth_2 = ground_truth_1.copy()
        predictions_2 = [
            RelevanceScore.IRRELEVANT,  # Perfect match
            RelevanceScore.PERFECTLY_RELEVANT,  # Off by 2
            RelevanceScore.HIGHLY_RELEVANT,  # Perfect match
            RelevanceScore.PERFECTLY_RELEVANT,  # Perfect match
        ] * 3

        result_1 = adapter.compute(ground_truth_1, predictions_1)
        result_2 = adapter.compute(ground_truth_2, predictions_2)

        # Larger ordinal distances should result in lower alpha
        assert result_2.value < result_1.value

    def test_empty_input_raises_error(self):
        """Test empty input raises ValueError."""
        adapter = KrippendorffsAlphaAdapter()

        with pytest.raises(ValueError, match="empty input"):
            adapter.compute([], [])

    def test_mismatched_lengths_raises_error(self):
        """Test mismatched input lengths raise ValueError."""
        adapter = KrippendorffsAlphaAdapter()
        ground_truth = [RelevanceScore.IRRELEVANT]
        predictions = [RelevanceScore.RELEVANT, RelevanceScore.HIGHLY_RELEVANT]

        with pytest.raises(ValueError, match="same length"):
            adapter.compute(ground_truth, predictions)

    def test_interpretation_thresholds(self):
        """Test interpretation categories match expected thresholds."""
        adapter = KrippendorffsAlphaAdapter()

        # Test each threshold category
        assert "poor" in adapter._interpret_alpha(-0.1)
        assert "insufficient" in adapter._interpret_alpha(0.5)
        assert "tentative" in adapter._interpret_alpha(0.75)
        assert "strong" in adapter._interpret_alpha(0.85)

    def test_data_formatting(self):
        """Test _format_reliability_data produces correct shape and values."""
        ground_truth = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT
        ]
        predictions = [
            RelevanceScore.RELEVANT,
            None,
            RelevanceScore.PERFECTLY_RELEVANT
        ]

        reliability_data = KrippendorffsAlphaAdapter._format_reliability_data(
            ground_truth, predictions
        )

        # Shape: 2 raters x 3 units
        assert reliability_data.shape == (2, 3)

        # Ground truth row (row 0)
        assert reliability_data[0, 0] == 0.0  # IRRELEVANT
        assert reliability_data[0, 1] == 1.0  # RELEVANT
        assert reliability_data[0, 2] == 2.0  # HIGHLY_RELEVANT

        # Predictions row (row 1)
        assert reliability_data[1, 0] == 1.0  # RELEVANT
        assert np.isnan(reliability_data[1, 1])  # None -> NaN
        assert reliability_data[1, 2] == 3.0  # PERFECTLY_RELEVANT

    def test_count_valid_pairs(self):
        """Test _count_valid_pairs counts non-None predictions correctly."""
        ground_truth = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT,
            RelevanceScore.PERFECTLY_RELEVANT
        ]
        predictions = [
            RelevanceScore.IRRELEVANT,
            None,
            RelevanceScore.HIGHLY_RELEVANT,
            None
        ]

        count = KrippendorffsAlphaAdapter._count_valid_pairs(ground_truth, predictions)

        assert count == 2

    def test_mixed_relevance_levels(self):
        """Test with mixed relevance levels to ensure ordinal scale works."""
        adapter = KrippendorffsAlphaAdapter()
        ground_truth = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT,
            RelevanceScore.PERFECTLY_RELEVANT,
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT
        ]
        predictions = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.HIGHLY_RELEVANT,  # Off by 1
            RelevanceScore.HIGHLY_RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT,  # Off by 1
            RelevanceScore.RELEVANT,  # Off by 1
            RelevanceScore.RELEVANT
        ]

        result = adapter.compute(ground_truth, predictions)

        # Should have some agreement but not perfect (alpha between 0 and 1)
        assert 0 <= result.value < 1
        assert result.sample_size == 6
        assert result.name == "krippendorffs_alpha"
        assert result.description is not None

    def test_partial_agreement_with_none(self):
        """Test partial agreement with some None values."""
        adapter = KrippendorffsAlphaAdapter()
        ground_truth = [
            RelevanceScore.IRRELEVANT,
            RelevanceScore.RELEVANT,
            RelevanceScore.HIGHLY_RELEVANT,
            RelevanceScore.PERFECTLY_RELEVANT,
            RelevanceScore.IRRELEVANT
        ]
        predictions = [
            RelevanceScore.IRRELEVANT,  # Match
            None,  # Missing
            RelevanceScore.RELEVANT,  # Off by 1
            None,  # Missing
            RelevanceScore.IRRELEVANT  # Match
        ]

        result = adapter.compute(ground_truth, predictions)

        # 3 valid predictions: 2 perfect matches, 1 close
        assert result.sample_size == 3
        # Should have high agreement (2/3 perfect, 1/3 close)
        assert result.value > 0
        assert isinstance(result.interpretation, str)
