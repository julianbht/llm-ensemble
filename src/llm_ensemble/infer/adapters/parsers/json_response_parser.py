"""JSON response parser adapter.

Parses JSON-formatted LLM judge outputs to extract relevance labels.
Supports both simple and multi-aspect formats.
"""

from __future__ import annotations
import json
import re

from llm_ensemble.infer.ports import ResponseParser
from llm_ensemble.infer.schemas.llm_score import LLMScore
from llm_ensemble.libs.schemas import RelevanceScore


class JsonResponseParser(ResponseParser):
    """Parser for JSON-formatted LLM responses.

    Expects JSON output in one of two formats:
    - Simple: {"O": 2}
    - Multi-aspect: {"M": 2, "T": 1, "O": 1}

    Where O is the final relevance score (0, 1, or 2).

    This implementation is based on the Thomas et al. prompt format but can
    work with any JSON format that includes an "O" field for the overall score.

    Example:
        >>> parser = JsonResponseParser()
        >>> label, warnings = parser.parse('{"O": 2}')
        >>> label
        2
        >>> label, warnings = parser.parse('The answer is {"O": 1} based on...')
        >>> label
        1
    """

    def __init__(self, score_field: str = "O"):
        """Initialize JSON response parser.

        Args:
            score_field: Name of the JSON field containing the relevance score (default: "O")
        """
        self.score_field = score_field

    def parse(self, raw_text: str) -> tuple[LLMScore, list[str]]:
        """Parse JSON response to extract relevance label.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (score, warnings):
            - score: LLMScore with extracted label (confidence/rationale may be None)
            - warnings: List of warning messages for parsing issues
        """
        warnings = []

        # Try to find JSON object in the response
        # Look for patterns like {"O": N} or {"M": N, "T": N, "O": N}
        json_pattern = r'\{[^}]*"' + self.score_field + r'"\s*:\s*\d+[^}]*\}'
        json_match = re.search(json_pattern, raw_text)

        if not json_match:
            warnings.append(f"No JSON object with '{self.score_field}' field found in response")
            return LLMScore(), warnings

        json_str = json_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            warnings.append(f"Failed to parse JSON: {e}")
            return LLMScore(), warnings

        # Extract the score
        score = data.get(self.score_field)

        if score is None:
            warnings.append(f"Missing '{self.score_field}' field in parsed JSON")
            return LLMScore(), warnings

        # Validate the score is 0, 1, or 2
        if not isinstance(score, int) or score not in [0, 1, 2, 3]:
            warnings.append(f"Invalid {self.score_field} score: {score} (expected 0, 1, 2, or 3)")
            return LLMScore(), warnings

        # Convert to RelevanceScore enum
        try:
            relevance_label = RelevanceScore(score)
        except ValueError:
            warnings.append(f"Invalid score value: {score}")
            return LLMScore(), warnings

        return LLMScore(label=relevance_label), warnings
