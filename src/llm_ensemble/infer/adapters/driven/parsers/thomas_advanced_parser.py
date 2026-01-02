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
from llm_ensemble.infer.domain.score_mappings import map_thomas_advanced_score
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class ThomasAdvancedParser(ForParsingResponses):
    """Parser for thomas-advanced prompt responses.

    Expects JSON output: {"M": N, "T": N, "O": N} where:
    - M: Match score (how well content matches query intent)
    - T: Trust score (trustworthiness of the web page)
    - O: Overall relevance score (0, 1, or 2)

    The "O" field is the final relevance score that we extract and map
    to the standard RelevanceScore scale.

    Mapping:
    - 0 (not relevant) → IRRELEVANT (0)
    - 1 (relevant, partly helpful) → RELEVANT (1)
    - 2 (highly relevant, very helpful) → HIGHLY_RELEVANT (2)
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

    def parse(self, raw_text: str) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Parse JSON response and create LLMScore domain entity.

        Extracts the "O" field from JSON response (with M, T, O fields)
        and constructs an LLMScore with extracted label. Returns None if no label could be extracted.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (LLMScore or None, parser_issue):
            - LLMScore: parsed fields, or None if no label could be extracted
            - parser_issue: Parser issue if encountered, None if successful
        """
        label: Optional[RelevanceScore] = None
        issue: Optional[ParserIssue] = None

        # Extract and validate score using testable helper methods
        json_data, issue = self._extract_json(raw_text)
        if json_data is not None:
            score_value, field_issue = self._extract_score_field(json_data)
            if field_issue:
                issue = field_issue
            if score_value is not None:
                label, validation_issue = self._validate_score(score_value)
                if validation_issue and not issue:
                    issue = validation_issue

        # Only create LLMScore if we successfully extracted a label
        if label is None:
            return None, issue

        score = LLMScore(
            label=label,
            confidence=None,
            rationale=None,
        )

        return score, None

    def get_parser(self) -> ResponseParser:
        """Get Parser metadata for this adapter.

        Returns:
            Parser entity with id and name
        """
        return self._parser

    def _extract_json(self, raw_text: str) -> tuple[Optional[dict], Optional[ParserIssue]]:
        """Extract JSON object with M, T, O fields from raw text.

        Pure, testable function for JSON extraction logic.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (parsed JSON dict or None, parser issue or None)
        """
        json_pattern = r'\{[^}]*"M"\s*:\s*\d+[^}]*"T"\s*:\s*\d+[^}]*"O"\s*:\s*\d+[^}]*\}'
        json_match = re.search(json_pattern, raw_text)

        if not json_match:
            issue = ParserIssue(
                code=ParserIssueCode.PARSE_ISSUE,
                message="No JSON object with 'M', 'T', 'O' fields found in response",
                metadata={"expected_format": '{"M": N, "T": N, "O": N}'}
            )
            return None, issue

        json_str = json_match.group(0)

        try:
            return json.loads(json_str), None
        except json.JSONDecodeError as e:
            issue = ParserIssue(
                code=ParserIssueCode.PARSE_ISSUE,
                message=f"Failed to parse JSON: {e}",
                metadata={"error_type": type(e).__name__}
            )
            return None, issue

    def _extract_score_field(self, json_data: dict) -> tuple[Optional[int], Optional[ParserIssue]]:
        """Extract the "O" score field from parsed JSON.

        Pure, testable function for field extraction logic.

        Args:
            json_data: Parsed JSON dict

        Returns:
            Tuple of (score value or None, parser issue or None)
        """
        score = json_data.get("O")

        if score is None:
            issue = ParserIssue(
                code=ParserIssueCode.FIELD_ISSUE,
                message="Missing 'O' field in parsed JSON",
                metadata={"field_name": "O"}
            )
            return None, issue

        return score, None

    def _validate_score(self, score_value: int) -> tuple[Optional[RelevanceScore], Optional[ParserIssue]]:
        """Validate score value and convert to RelevanceScore enum.

        Pure, testable function for validation logic.
        Validates score is in thomas-advanced range (0-2) then maps to standard scale
        using domain mapping function.

        Args:
            score_value: Raw score value from JSON (expected 0, 1, or 2)

        Returns:
            Tuple of (RelevanceScore enum or None, parser issue or None)
        """
        if not isinstance(score_value, int) or score_value not in [0, 1, 2]:
            issue = ParserIssue(
                code=ParserIssueCode.VALIDATION_ISSUE,
                message=f"Invalid O score: {score_value} (expected 0, 1, or 2)",
                metadata={"field_name": "O", "actual_value": str(score_value)}
            )
            return None, issue

        return map_thomas_advanced_score(score_value), None
