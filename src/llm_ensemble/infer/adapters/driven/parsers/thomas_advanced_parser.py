"""Thomas et al. advanced prompt parser.

Parses JSON responses from the thomas-advanced prompt which outputs {"M": N, "T": N, "O": N} format.
Parser knows exactly what to look for - tightly coupled to the thomas-advanced prompt.
"""

from __future__ import annotations
import json
import re
import uuid
from typing import Optional

from llm_ensemble.infer.application.ports.driven.for_parsing_responses import ForParsingResponses
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssue, ParserIssueCode
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class ThomasAdvancedParser(ForParsingResponses):
    """Parser for thomas-advanced prompt responses.

    Expects JSON output: {"M": N, "T": N, "O": N} where:
    - M: Match score (how well content matches query intent)
    - T: Trust score (trustworthiness of the web page)
    - O: Overall relevance score (0, 1, 2, or 3)

    The "O" field is the final relevance score that we extract.
    """

    PARSER_NAME = "thomas-advanced"
    PARSER_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "llm-ensemble.thomas-advanced-parser-v1")

    def __init__(self):
        """Initialize parser and create cached parser entity."""
        self._parser = ResponseParser(
            id=self.PARSER_ID,
            name=self.PARSER_NAME,
            version="1.0"
        )

    def parse(self, raw_text: str) -> tuple[Optional[LLMScore], list[ParserIssue]]:
        """Parse JSON response and create LLMScore domain entity.

        Extracts the "O" field from JSON response (with M, T, O fields)
        and constructs an LLMScore with extracted label. Returns None if no label could be extracted.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (LLMScore or None, warnings):
            - LLMScore: parsed fields, or None if no label could be extracted
            - warnings: List of parser warnings from the parsing process
        """
        warnings: list[ParserIssue] = []
        label: Optional[RelevanceScore] = None

        # Extract and validate score using testable helper methods
        json_data = self._extract_json(raw_text, warnings)
        if json_data is not None:
            score_value = self._extract_score_field(json_data, warnings)
            if score_value is not None:
                label = self._validate_score(score_value, warnings)

        # Only create LLMScore if we successfully extracted a label
        if label is None:
            return None, warnings

        score = LLMScore(
            label=label,
            confidence=None,
            rationale=None,
        )

        return score, warnings

    def get_parser(self) -> ResponseParser:
        """Get Parser metadata for this adapter.

        Returns:
            Parser entity with id and name
        """
        return self._parser

    def _extract_json(self, raw_text: str, warnings: list[ParserIssue]) -> Optional[dict]:
        """Extract JSON object with M, T, O fields from raw text.

        Pure, testable function for JSON extraction logic.

        Args:
            raw_text: Raw text response from the LLM
            warnings: List to append warnings to

        Returns:
            Parsed JSON dict if successful, None otherwise
        """
        json_pattern = r'\{[^}]*"M"\s*:\s*\d+[^}]*"T"\s*:\s*\d+[^}]*"O"\s*:\s*\d+[^}]*\}'
        json_match = re.search(json_pattern, raw_text)

        if not json_match:
            warning = ParserIssue(
                code=ParserIssueCode.PARSE_ERROR,
                message="No JSON object with 'M', 'T', 'O' fields found in response",
                metadata={"expected_format": '{"M": N, "T": N, "O": N}'}
            )
            warnings.append(warning)
            return None

        json_str = json_match.group(0)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            warning = ParserIssue(
                code=ParserIssueCode.PARSE_ERROR,
                message=f"Failed to parse JSON: {e}",
                metadata={"error_type": type(e).__name__}
            )
            warnings.append(warning)
            return None

    def _extract_score_field(self, json_data: dict, warnings: list[ParserIssue]) -> Optional[int]:
        """Extract the "O" score field from parsed JSON.

        Pure, testable function for field extraction logic.

        Args:
            json_data: Parsed JSON dict
            warnings: List to append warnings to

        Returns:
            Score value if present, None otherwise
        """
        score = json_data.get("O")

        if score is None:
            warning = ParserIssue(
                code=ParserIssueCode.FIELD_ERROR,
                message="Missing 'O' field in parsed JSON",
                metadata={"field_name": "O"}
            )
            warnings.append(warning)
            return None

        return score

    def _validate_score(self, score_value: int, warnings: list[ParserIssue]) -> Optional[RelevanceScore]:
        """Validate score value and convert to RelevanceScore enum.

        Pure, testable function for validation logic.

        Args:
            score_value: Raw score value from JSON
            warnings: List to append warnings to

        Returns:
            RelevanceScore enum if valid, None otherwise
        """
        if not isinstance(score_value, int) or score_value not in [0, 1, 2, 3]:
            warning = ParserIssue(
                code=ParserIssueCode.VALIDATION_ERROR,
                message=f"Invalid O score: {score_value} (expected 0, 1, 2, or 3)",
                metadata={"field_name": "O", "actual_value": str(score_value)}
            )
            warnings.append(warning)
            return None

        try:
            return RelevanceScore(score_value)
        except ValueError:
            warning = ParserIssue(
                code=ParserIssueCode.VALIDATION_ERROR,
                message=f"Invalid score value: {score_value}",
                metadata={"value": score_value}
            )
            warnings.append(warning)
            return None
