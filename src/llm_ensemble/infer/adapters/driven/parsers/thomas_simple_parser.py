"""Thomas et al. simple prompt parser.

Parses JSON responses from the thomas-simple prompt which outputs {"score": N} format.
Uses cascading strategy for robust parsing of LLM responses.
Supports 4-level TREC scale (0-3).
"""

from __future__ import annotations
import json
import re
import uuid
from typing import Any, Optional

from llm_ensemble.infer.application.ports.driven.for_parsing_responses import ForParsingResponses
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssue, ParserIssueCode
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class ThomasSimpleParser(ForParsingResponses):
    """Parser for thomas-simple prompt responses.

    Expects JSON output: {"score": N} where N is 0-3.

    Uses cascading strategy:
    1. Try clean JSON parsing
    2. Try embedded JSON extraction (markdown blocks, text)
    3. Try regex pattern matching
    4. Try fuzzy text matching

    Supports 4-level TREC scale:
    - 0 = Irrelevant
    - 1 = Relevant
    - 2 = Highly Relevant
    - 3 = Perfectly Relevant
    """

    PARSER_NAME = "thomas-simple"
    PARSER_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "llm-ensemble.thomas-simple-parser-v1")

    def __init__(self):
        """Initialize parser and create cached parser entity."""
        self._parser = ResponseParser(
            id=self.PARSER_ID,
            name=self.PARSER_NAME,
            version="1.0"
        )

    def parse(self, raw_text: str) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Parse JSON response using cascading strategy.

        Tries multiple extraction strategies in order of reliability.
        Returns None if all strategies fail.
        Stops at first successful extraction (early return).

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (LLMScore or None, parser_issue):
            - LLMScore: parsed score, or None if extraction failed
            - parser_issue: Primary parser issue if encountered, None if clean parse
        """
        # Cascading strategy: try each method until one succeeds
        strategies = [
            self._try_clean_json,
            self._try_embedded_json,
            self._try_regex_extract,
            self._try_fuzzy_extract,
        ]

        for strategy in strategies:
            score, issue = strategy(raw_text)
            if score is not None:
                return score, issue

        # All strategies failed
        return None, ParserIssue(
            code=ParserIssueCode.PARSE_FAILED,
            message="All parsing strategies failed to extract score",
            metadata={"strategies_tried": len(strategies)}
        )

    def get_parser(self) -> ResponseParser:
        """Get Parser metadata for this adapter.

        Returns:
            Parser entity with id and name
        """
        return self._parser

    # ========================================================================
    # Cascading parsing strategies
    # ========================================================================

    def _try_clean_json(self, raw_text: str) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Strategy 1: Try parsing response as clean JSON at root level.

        Args:
            raw_text: Raw LLM response

        Returns:
            Tuple of (LLMScore or None, parser_issue)
        """
        try:
            data = json.loads(raw_text.strip())
            if isinstance(data, dict):
                return self._extract_score_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass
        return None, None

    def _try_embedded_json(self, raw_text: str) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Strategy 2: Try extracting JSON from markdown blocks or embedded text.

        Looks for:
        - ```json ... ```
        - ``` ... ```
        - JSON objects embedded in explanation text

        Args:
            raw_text: Raw LLM response

        Returns:
            Tuple of (LLMScore or None, parser_issue)
        """
        # Try markdown code blocks first
        markdown_patterns = [
            r'```json\s*(\{[^`]+\})\s*```',
            r'```\s*(\{[^`]+\})\s*```',
        ]

        for pattern in markdown_patterns:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict):
                        score, _ = self._extract_score_from_dict(data)
                        if score:
                            return score, ParserIssue(
                                code=ParserIssueCode.NON_STANDARD_FORMAT,
                                message="Extracted JSON from markdown code block",
                                metadata={"extraction_method": "markdown"}
                            )
                except (json.JSONDecodeError, ValueError):
                    continue

        # Try finding JSON object with score field anywhere in text
        json_pattern = r'\{[^}]*"score"\s*:\s*\d+[^}]*\}'
        match = re.search(json_pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    score, _ = self._extract_score_from_dict(data)
                    if score:
                        return score, ParserIssue(
                            code=ParserIssueCode.NON_STANDARD_FORMAT,
                            message="Extracted JSON object from text",
                            metadata={"extraction_method": "embedded_json"}
                        )
            except (json.JSONDecodeError, ValueError):
                pass

        return None, None

    def _try_regex_extract(self, raw_text: str) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Strategy 3: Try extracting score using regex patterns.

        Looks for patterns like:
        - "score": 2
        - score: 2
        - "score" = 2

        Args:
            raw_text: Raw LLM response

        Returns:
            Tuple of (LLMScore or None, parser_issue)
        """
        patterns = [
            r'"score"\s*:\s*(\d+)',
            r'score\s*:\s*(\d+)',
            r'"score"\s*=\s*(\d+)',
            r'score\s*=\s*(\d+)',
            # Also try standalone numbers at end of response
            r'\b([0-3])\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                try:
                    score_value = int(match.group(1))
                    label = self._validate_score_value(score_value)
                    if label is not None:
                        return LLMScore(label=label, confidence=None, rationale=None), ParserIssue(
                            code=ParserIssueCode.NON_STANDARD_FORMAT,
                            message="Extracted score using regex pattern matching",
                            metadata={"extraction_method": "regex", "pattern": pattern}
                        )
                except ValueError:
                    continue

        return None, None

    def _try_fuzzy_extract(self, raw_text: str) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Strategy 4: Try fuzzy matching based on keywords.

        Last resort strategy looking for relevance keywords in text.

        Args:
            raw_text: Raw LLM response

        Returns:
            Tuple of (LLMScore or None, parser_issue)
        """
        text_lower = raw_text.lower()

        issue = ParserIssue(
            code=ParserIssueCode.LOW_CONFIDENCE_EXTRACTION,
            message="Extracted score using fuzzy keyword matching",
            metadata={"extraction_method": "fuzzy", "confidence": "low"}
        )

        # Check for explicit relevance keywords
        if any(word in text_lower for word in ["perfectly relevant", "perfect match", "exact answer"]):
            return LLMScore(label=RelevanceScore.PERFECTLY_RELEVANT, confidence=None, rationale=None), issue

        if any(word in text_lower for word in ["highly relevant", "very helpful", "vital information"]):
            return LLMScore(label=RelevanceScore.HIGHLY_RELEVANT, confidence=None, rationale=None), issue

        if any(word in text_lower for word in ["related", "partly helpful"]):
            return LLMScore(label=RelevanceScore.RELEVANT, confidence=None, rationale=None), issue

        if any(word in text_lower for word in ["not relevant", "irrelevant", "nothing to do"]):
            return LLMScore(label=RelevanceScore.IRRELEVANT, confidence=None, rationale=None), issue

        return None, None

    # ========================================================================
    # Helper methods for extraction and validation
    # ========================================================================

    def _extract_score_from_dict(self, data: dict[str, Any]) -> tuple[Optional[LLMScore], Optional[ParserIssue]]:
        """Extract and validate score field from parsed JSON dict.

        Args:
            data: Parsed JSON dictionary

        Returns:
            Tuple of (LLMScore or None, parser_issue)
        """
        # Try common field names
        score_value = None
        for field_name in ["score", "Score", "SCORE", "relevance", "Relevance"]:
            if field_name in data:
                score_value = data[field_name]
                break

        if score_value is None:
            return None, ParserIssue(
                code=ParserIssueCode.MISSING_REQUIRED_FIELD,
                message="Missing 'score' field in parsed JSON",
                metadata={"field_name": "score", "available_fields": list(data.keys())}
            )

        label = self._validate_score_value(score_value)
        if label is None:
            return None, ParserIssue(
                code=ParserIssueCode.INVALID_FIELD_VALUE,
                message=f"Invalid score value: {score_value}",
                metadata={"field_name": "score", "actual_value": str(score_value)}
            )

        return LLMScore(label=label, confidence=None, rationale=None), None

    def _validate_score_value(self, score_value: Any) -> Optional[RelevanceScore]:
        """Validate score value and convert to RelevanceScore enum.

        Handles range validation only (no type coercion).
        Valid range: 0-3 (4-level TREC scale).

        Args:
            score_value: Raw score value from JSON or text

        Returns:
            RelevanceScore enum if valid, None otherwise
        """
        # Handle string scores
        if isinstance(score_value, str):
            try:
                score_value = int(score_value)
            except ValueError:
                return None

        # Validate type and range
        if not isinstance(score_value, int) or score_value not in [0, 1, 2, 3]:
            return None

        try:
            return RelevanceScore(score_value)
        except ValueError:
            return None
